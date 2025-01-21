from utils.mydataloader import MyDataloader
from opt import OptInit

if __name__ == "__main__":
    settings = OptInit(dataset="ABIDE", atlas="aal")
    opt = settings.initialize()
    dl = MyDataloader(opt)
    abide = dl.fetch_abide("rois_aal")
    fcs = dl.process_abide()
