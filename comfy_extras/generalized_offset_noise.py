import torch

class GeneralizedOffsetNoise:
    """
    Implements the inference-side logic for the "Generalized Diffusion Model with Adjusted Offset Noise"
    (https://arxiv.org/abs/2412.03134v1).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {
                    "default": 0.100,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.001,
                    "display": "number"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/sampling"

    def patch(self, model, strength, seed):
        if strength == 0:
            return (model,)

        is_first_step = [True]

        def first_conv_patch(x, transformer_options):
            if is_first_step[0]:
                is_first_step[0] = False

                generator = torch.Generator(device=x.device).manual_seed(seed)

                offset_noise_shape = (x.shape[0], x.shape[1], 1, 1)
                offset_noise = torch.randn(
                    offset_noise_shape,
                    generator=generator,
                    device=x.device,
                    dtype=x.dtype
                ) * strength

                print(f"Applying Generalized Offset Noise with strength {strength} on first step.")

                return x + offset_noise

            return x

        model_clone = model.clone()
        model_clone.set_model_input_block_patch(first_conv_patch)

        return (model_clone,)

NODE_CLASS_MAPPINGS = {
    "GeneralizedOffsetNoise": GeneralizedOffsetNoise
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeneralizedOffsetNoise": "Generalized Offset Noise (Inference)"
}