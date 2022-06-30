from train.gradient import requires_grad
from train.gradient_manager import GradientManager
from train.coach import Coach
from train.flow_coach import FlowCoach
from train.flow_coach_random_samples import RandomFlowCoach
from train.fixed_wplus_flow_coach import WPlusFlowCoach # remove fixed
from train.txt2img_mapping_coach_fixed import Txt2ImgFlowCoach
# from train.txt2img_mapping_coach import Txt2ImgFlowCoach
from train.txt2w_mapping_coach import Text2WFlowCoach
from .txt2place_coach import Text2PlaceCoach
