from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
# from few_shot_learning_system import MAMLFewShotClassifier
from few_shot_learning_system_context import MAMLFewShotClassifier
# from few_shot_learning_system_class_aware_context import MAMLFewShotClassifier
# from few_shot_learning_system_aMAML import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset
from multiprocessing import freeze_support

if __name__ == '__main__':

    # python train_maml_system.py --name_of_args_json_file experiment_config/f-maml_miniImagenet.json --gpu_to_use 0
    # python train_maml_system.py --name_of_args_json_file experiment_config/a-maml_miniImagenet.json --gpu_to_use 0
    # python train_maml_system.py --name_of_args_json_file experiment_config/HyperGenerator_miniImagenet.json --gpu_to_use 0
    # python train_maml_system.py --name_of_args_json_file experiment_config/HyperAutoEncoder_miniImagenet.json --gpu_to_use 0
    # python train_maml_system.py --name_of_args_json_file experiment_config/HyperGNN_miniImagenet.json --gpu_to_use 0

    freeze_support()

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, args.image_channels,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)
    data = MetaLearningSystemDataLoader
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
