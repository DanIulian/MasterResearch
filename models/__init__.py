from . import a3c_cnn_policy
from . import atari_cnn_policy
from . import mlp_policy
""" Each model script should have the method get_models() which returns a list of models """

POLICY_MODELS = {
    "mlp_policy":
    mlp_policy,
    "atari_cnn":
    atari_cnn_policy,
    "a3c_cnn":
    a3c_cnn_policy
}


def get_models(cfg):

    # @name         : name of the model
    assert hasattr(
        cfg, "policy"
    ) and cfg.policy in POLICY_MODELS, "Please provide a valid model name."

    return POLICY_MODELS[cfg.policy].get_models()

