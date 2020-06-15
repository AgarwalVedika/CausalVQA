#Imports related to object removal
import torch
import torch.nn as nn
from solver_augment import Solver
from torch.autograd import Variable
import torch.nn.functional as F

class ParamObject(object):

    def __init__(self, adict):
        """Convert a dictionary to a class

        @param :adict Dictionary
        """

        self.__dict__.update(adict)

        for k, v in adict.items():
            if isinstance(v, dict):
                self.__dict__[k] = ParamObject(v)

    def __getitem__(self,key):
        return self.__dict__[key]

    def values(self):
        return self.__dict__.values()

    def itemsAsDict(self):
        return dict(self.__dict__.items())


class ObjectRemover(nn.Module):

    def __init__(self, removal_model = None, no_inpainter = 0, dilateMask = 11, pixel_mean=[0,0,0], remover_type='rand', no_remover=0):
        super(ObjectRemover, self).__init__()
        removal_model = '/BS/rshetty-wrk/work/code/controlled-generation/trained_models_2/stargan/fulleditor/checkpoint_stargan_coco_fulleditor_wgan_50pcUnion_msz32_withPmask_L1150_style3k_tv_nb4_512sz_sqznet_mgpu_b14_255_3483.pth.tar' if removal_model is None else removal_model
        self.no_inpainter = no_inpainter
        self.no_remover = no_remover
        self.avg_inpainter = 0
        self.remover_type = remover_type
        self.ext_means = nn.Parameter(torch.FloatTensor(pixel_mean).view(1,-1,1,1),requires_grad=False)
        self.dilateWeight = None
        if not self.no_remover:
            model = torch.load(removal_model)
            model['generator_state_dict'] = {k:model['generator_state_dict'][k] for k in model['generator_state_dict'] if 'running_' not in k}
            r_configs = model['arch']
            r_configs['pretrained_model'] = removal_model
            r_configs['load_encoder'] = 0
            r_configs['no_inpainter'] = no_inpainter
            r_configs['load_discriminator'] = 0
            r_configs['lowres_mask'] = 0
            self.r_config = r_configs
            self.r_solvers = Solver(None, None, ParamObject(r_configs), mode='inpainting', pretrainedcv=model)
            self.r_solvers.G.eval()
            self.G = self.r_solvers.G
            for p in self.G.parameters():
                p.requires_grad = False

            if dilateMask:
                self.dilateWeight = torch.ones((1,1,dilateMask,dilateMask))
                self.dilateWeight = nn.Parameter(self.dilateWeight,requires_grad=False)
            else:
                self.dilateWeight = None

    def flip(self,img,axis=1):
        with torch.no_grad():
            axlen = img.size(axis)
            axindex = torch.arange(axlen-1,-1,-1,dtype=torch.long,device=img.device)
            return img.index_select(axis, axindex)

    def convertImgToCentered(self, img):
        with torch.no_grad():
            # Assuming th input image is natural image with some mean subtracted and in BGR format.
            # Normalized it between 0 and 1, make it zero mean, and unit variance and RGB format
            return ((self.flip(img + self.ext_means,1)/256.) - 0.5)/0.5

    def convertCenteredToImg(self, img):
        with torch.no_grad():
            return (((self.flip(img,1) + 1.)*256./2.) - self.ext_means)

    def getImageSizeMask(self, mask, img_size=None):
        #return F.upsample(mask, scale_factor=int(self.image_size/mask.size(-1)), mode=self.m_upsample_type) if mask.size(-1)!= self.image_size else mask
        return F.adaptive_max_pool2d(mask, img_size)

    def forward_inpainter(self, x, label, get_feat=False, binary_mask=None, onlyMasks=False, mask_threshold=0.3,
                           gtMask=None, withGTMask=False, dilate=None, getAllMasks=False, n_iter = 0):

        # extract features from the box
        if withGTMask:
            mask = gtMask
            boxFeat = None
            allMasks = None

        nnzMask = mask.sum(dim=2).sum(dim=2).view(-1).nonzero().view(-1)

        if len(nnzMask) and not self.no_remover:
            xNNz = x[nnzMask]
            maskUpsamp = self.getImageSizeMask(mask[nnzMask], [xNNz.size(-2), xNNz.size(-1)])# if (x.size(-1) != mask.size(-1)) or (x.size(-2) != mask.size(-2)) else mask
            if dilate is not None:

                dsz = dilate.size(-1)//2
                maskUpsamp = torch.clamp(F.conv2d(F.pad(maskUpsamp,(dsz,dsz,dsz,dsz)),dilate), max=1.0, min=0.0)
                maskUpsamp = maskUpsamp[:,:,0:x.size()[2],0:x.size()[3]]  #TODO quick_fix vedika
                mask = maskUpsamp

            xM = (1-maskUpsamp)*xNNz

            # Pass the boxfeature and masked image to the
            if self.no_inpainter:
                fakeImg = xM
                feat = None
            elif self.avg_inpainter:
                fakeImg = xM + maskUpsamp* (maskUpsamp*xNNz).mean()
                feat = None
            else:
                #mask the input image and append the mask as a channel
                xInp =torch.cat([xM,maskUpsamp],dim=1)
                genImg = self.G(xInp, boxFeat)
                feat = None
                fakeImg = genImg*maskUpsamp+ xM

            outImg = x.clone()
            outImg[nnzMask] = fakeImg
        else:
            outImg = x
            feat = None

        if getAllMasks:
            return outImg, feat, mask, allMasks
        else:
            return outImg, feat, mask


    def forward(self, img, mask):
        with torch.no_grad():
            #print('object remocver',img.device, self.ext_means.device)
            centeredImg = img
            fake_x, _, mask_out = self.forward_inpainter(centeredImg, None, gtMask = mask, withGTMask=True, dilate = self.dilateWeight)
            #fake_x = centeredImg#, _, mask_out = self.r_solvers.forward_fulleditor(centeredImg, None, gtMask = mask, withGTMask=True, dilate = self.dilateWeight)
            return fake_x





