from transformers import PretrainedConfig

class Prompt_Fusion_Config(PretrainedConfig):
    model_type = "prompt_fusion"

    def __init__(
        self, 
        model_type='prompt_fusion',
        num_hidden_layers=24, 
        num_hidden_prompt_layers=12,
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=512,
        relative_attention=True,
        position_buckets=256,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        pos_att_type="p2c|c2p",
        layer_norm_eps=1e-7,
        max_relative_positions=-1,
        position_biased_input=False,
        num_attention_heads=16,
        type_vocab_size=0,
        vocab_size=128100,
        **kwargs
    ):
        super().__init__(**kwargs)
        # General model configuration
        self.model_type = model_type
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.relative_attention = relative_attention
        self.position_buckets = position_buckets
        self.norm_rel_ebd = norm_rel_ebd
        self.share_att_key = share_att_key
        self.pos_att_type = pos_att_type
        self.layer_norm_eps = layer_norm_eps
        self.max_relative_positions = max_relative_positions
        self.position_biased_input = position_biased_input
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.num_hidden_prompt_layers = num_hidden_prompt_layers
