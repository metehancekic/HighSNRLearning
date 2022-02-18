from .attention_layers import ScaledDotProductAttention, VanillaMLPAttention, VanillaConvolutionalAttention, FeatureAttention, ConvolutionalFeatureAttention, SpatialAttention
from .can_tools import take_top_coeff, take_top_coeff_BPDA, take_top_k
# from .center_surround import CenterSurroundModule, CenterSurroundConv, DoGLayer, DoGLowpassLayer, LowpassLayer, DoG_LP_Layer
from .combiner import Combined
from .datanorm import Normalize
from .decoders import Decoder
from .divisive_normalization import DivisiveNormalization2d
from .gabor import GaborConv2d
from .gaussian import GaussianConv2d, DifferenceOfGaussianConv2d
from .lowpass import LowPassConv2d
from .channel_pool import ChannelPool
from .lp_norm_conv import LpConv2d
from .binarization import Binarization

# from .frontends import LP_Gabor_Layer, LP_Gabor_Layer_v2, LP_Gabor_Layer_v3, LP_Gabor_Layer_v4, LP_Gabor_Layer_v5,  LP_Gabor_Layer_v6, LP_Layer, Identity, Shape_Layer
