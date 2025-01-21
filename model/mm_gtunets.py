import numpy as np
import torch
from torch_geometric.utils import dropout_edge, dense_to_sparse
import torch.nn.functional as F
import torch.nn as nn
from utils.tools import cal_feature_sim, EarlyStopping
from model.gtunet import GTUNet
import os


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.input_dim = len(args.scores)
        self.hidden_dim = 256
        self.latent_dim = args.node_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            # output mean and variance
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            # used to process input data in the range [0, 1]
            nn.Sigmoid()
        )
        # the reconstruction loss function uses mean square error
        self.criterion = nn.MSELoss(reduction='sum')

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :self.latent_dim]
        logvar = mu_logvar[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        # calculate the reconstruction loss
        reconstruction_loss = self.criterion(recon_x, x)
        # calculating KL divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = reconstruction_loss + kl_divergence
        return z, total_loss


class CustomConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2d, self).__init__(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.weight.data = torch.abs(self.weight.data)
        self.weight.data[:, 1, :, :] = - self.weight.data[:, 1, :, :]

    def forward(self, x):
        # Execute custom operation during forward pass
        self.weight.data[:, 1, :, :] = -torch.max(torch.abs(self.weight.data[:, 1, :, :]),
                                                  self.weight.data[:, 1, :, :] + self.weight.data[:, 2, :, :])
        self.weight.data[:, 0, :, :] = torch.max(self.weight.data[:, 0, :, :], self.weight.data[:, 2, :, :])
        return super(CustomConv2d, self).forward(x)


class RP_Attention(nn.Module):
    def __init__(self, args):
        super(RP_Attention, self).__init__()
        self.args = args
        self.num_score = len(args.scores)
        self.device = args.device

        # weight parameters for phenotypic data
        self.weights = nn.Parameter(torch.rand(self.num_score)).to(self.device)
        self.weights.data /= self.weights.data.sum()

        # reward, penalty and incentive coefficients
        self.conv = CustomConv2d(in_channels=3, out_channels=1)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_score):
            self.convs.append(self.conv)

    def forward(self, affinity_graphs):
        outputs = []
        value = []

        for i in range(self.num_score):
            outputs.append(F.sigmoid(self.convs[i](affinity_graphs[i]).squeeze()))
            reward = self.convs[i].weight.data[:, 0, :, :].squeeze()
            penalty = self.convs[i].weight.data[:, 1, :, :].squeeze()
            value.append(self.cal_value(affinity_graphs[i], reward, penalty))

        outputs = torch.stack(outputs)
        rp_graph = (self.weights.view(self.num_score, 1, 1) * outputs).sum(dim=0)

        value = torch.tensor(value).sum()

        return rp_graph, value

    def cal_value(self, graphs, reward, penalty):
        reward_graph = graphs[0, :, :]
        penalty_graph = graphs[1, :, :]

        value = torch.sum(F.relu(reward * reward_graph + penalty * penalty_graph)) / (self.args.num_subjects ** 2)
        return value


class Multimodal_Attention(nn.Module):
    def __init__(self, args):
        super(Multimodal_Attention, self).__init__()
        self.channel = args.out
        self.shared_lin = nn.Linear(self.channel, self.channel)
        self.img_lin = nn.Linear(self.channel, self.channel)
        self.ph_lin = nn.Linear(self.channel, self.channel)

    def cal_attention_score(self, attention, shared_attention):
        attention_score = torch.trace(torch.mm(attention, attention.t())) / torch.trace(
            torch.mm(shared_attention, shared_attention.t()))
        return attention_score

    def forward(self, img_embed, ph_embed):
        shared_embed = 0.5 * (img_embed + ph_embed)

        img_attention = F.tanh(self.img_lin(img_embed))
        ph_attention = F.tanh(self.ph_lin(ph_embed))
        shared_attention = F.tanh(self.shared_lin(shared_embed))

        # modality weight calculation
        img_shared_score = self.cal_attention_score(img_attention, shared_attention)
        ph_shared_score = self.cal_attention_score(ph_attention, shared_attention)
        attention_scores = F.softmax(torch.tensor([img_shared_score, ph_shared_score]), dim=0)
        img_weight = attention_scores[0]
        ph_weight = attention_scores[1]

        # modality joint Representation
        joint_embed = shared_attention * shared_embed + img_attention * img_embed + ph_attention * ph_embed

        return joint_embed, img_weight, ph_weight


class MM_GTUNets(nn.Module):
    def __init__(self, args, fold):
        super(MM_GTUNets, self).__init__()
        self.args = args
        self.fold = fold
        self.device = args.device
        self.dropout = args.dropout
        self.edge_drop = args.edge_drop
        self.load_pretrain = True
        self.setup_vae_pretrain()
        self.setup_attenton()

        self.img_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                               depth=args.img_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)
        self.ph_unet = GTUNet(in_channels=args.node_dim, hidden_channels=args.hidden, out_channels=args.out,
                              depth=args.ph_depth, edge_dim=1, pool_ratios=args.pool_ratios, dropout=args.dropout)

        self.clf = nn.Sequential(
            torch.nn.Linear(args.out, 256),
            torch.nn.ReLU(),
            nn.BatchNorm1d(256),
            torch.nn.Linear(256, args.num_classes))

    def setup_vae_pretrain(self):
        self.vae = VAE(self.args)
        self.init_vae_optimizer()
        self.init_vae_save_path()
        self.init_vae_early_stop()

    def setup_attenton(self):
        self.rp_attention = RP_Attention(self.args)
        self.mm_attention = Multimodal_Attention(self.args)

    def init_vae_save_path(self):
        self.vae_save_path = self.args.ckpt_path + "/fold{}_pretrain.pth".format(self.fold)

    def init_vae_optimizer(self):
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.args.vae_lr,
                                              weight_decay=5e-4)

    def init_vae_early_stop(self):
        self.early_stopping = EarlyStopping(patience=self.args.early_stop, verbose=True)

    def load_vae(self):
        # loading pre-trained parameters
        self.vae.load_state_dict(torch.load(self.vae_save_path, map_location=self.device))

        # freeze the parameters
        for param in self.vae.parameters():
            param.requires_grad = False

    def train_vae(self, ph_features):
        print("Start pretraining vae...")
        best_loss = 1e50
        best_epo = 0
        ph_features = ph_features.to(self.device)
        for epoch in range(3000):
            self.vae.train()
            self.vae_optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                _, loss = self.vae(ph_features)

            loss.backward()
            self.vae_optimizer.step()

            if epoch % 100 == 0:
                print(
                    "Epoch: {},\tlr: {:.5f},\tloss: {:.5f}".format(epoch, self.vae_optimizer.param_groups[0]['lr'],
                                                                   loss.item()))
            if best_loss > loss:
                best_loss = loss
                best_epo = epoch
                if (self.args.ckpt_path != ''):
                    if not os.path.exists(self.args.ckpt_path):
                        os.makedirs(self.args.ckpt_path)
                    torch.save(self.vae.state_dict(), self.vae_save_path)
                    # print("Epoch:{} {} Saved vae to:{}".format(epoch, "\u2714", self.vae_save_path))

            self.early_stopping(loss, self.vae)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        print("\r\n => Fold {} best pretrain vae loss {:.5f}, epoch {}\n".format(self.fold, best_loss, best_epo))

    def create_rp_graph(self, affinity_graphs):
        rp_graph, value = self.rp_attention(affinity_graphs)

        return rp_graph, value

    def cal_graph_loss(self, img_embed, ph_embed):
        L = torch.diagflat(torch.sum(self.fused_graph, -1)) - self.fused_graph
        # smoothness loss
        img_smh_loss = self.cal_smh_loss(img_embed, L)
        ph_smh_loss = self.cal_smh_loss(ph_embed, L)

        # degree loss
        deg_loss = self.cal_deg_loss()

        # graph regularization
        img_loss = self.args.smh * img_smh_loss + self.args.deg * deg_loss
        ph_loss = self.args.smh * ph_smh_loss + self.args.deg * deg_loss

        # reward regularization
        reward_loss = self.cal_reward_loss()

        graph_loss = self.img_weight * img_loss + self.ph_weight * (ph_loss + reward_loss)

        return graph_loss

    def cal_smh_loss(self, embed, L):
        smh_loss = torch.trace(torch.mm(embed.T, torch.mm(L, embed)) / torch.prod(
            torch.tensor(self.fused_graph.shape, dtype=torch.float)))
        return smh_loss

    def cal_deg_loss(self):
        one = torch.ones(self.fused_graph.size(-1)).to(self.device)
        deg_loss = torch.sum(torch.mm(self.fused_graph, one.unsqueeze(-1) + 1e-5).log()) / self.fused_graph.shape[
            -1]
        return deg_loss

    def cal_reward_loss(self):
        reward_loss = self.args.val * (1 / (self.value + 1e-5))
        return reward_loss

    def forward(self, img_features, ph_features, affinity_graphs):
        img_features = img_features.to(self.device)
        ph_features = ph_features.to(self.device)
        affinity_graphs = affinity_graphs.to(self.device)

        # reconstruction of phenotypic features
        if self.load_pretrain:
            self.load_vae()
        ph_features, _ = self.vae(ph_features)

        # construct the affinity graph
        self.rp_graph, self.value = self.create_rp_graph(affinity_graphs)

        # feature fusion and calculate the similarity of fused multimodal features
        fused_embed = torch.cat((img_features, ph_features), dim=1)
        self.fused_sim = cal_feature_sim(fused_embed)

        # compute adaptive reward population graph
        self.fused_graph = self.fused_sim * self.rp_graph
        fused_index, fused_attr = dense_to_sparse(self.fused_graph)
        fused_attr = fused_attr.view(-1, 1)

        # edge dropout
        if self.training and (self.edge_drop > 0):
            fused_index, fused_mask = dropout_edge(fused_index)
            fused_attr = fused_attr[fused_mask]

        img_embed = self.img_unet(img_features, fused_index, fused_attr)
        ph_embed = self.ph_unet(ph_features, fused_index, fused_attr)

        # obtain modality-joint representation and update weights of imaging and phenotypic data
        self.joint_embed, self.img_weight, self.ph_weight = self.mm_attention(img_embed, ph_embed)

        # computing graph loss
        graph_loss = self.cal_graph_loss(img_embed, ph_embed)

        outputs = self.clf(self.joint_embed)

        return outputs, graph_loss
