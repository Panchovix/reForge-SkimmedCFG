import logging
import gradio as gr
from modules import scripts
from skimmed_CFG.skimmed_CFG import CFG_skimming_single_scale_pre_cfg_node, skimReplacePreCFGNode, SkimmedCFGLinInterpCFGPreCFGNode, SkimmedCFGLinInterpDualScalesCFGPreCFGNode

class SkimmedCFGScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.mode = "Single Scale"
        self.skimming_cfg = 7.0
        self.full_skim_negative = False
        self.disable_flipping_filter = False
        self.lin_interp_cfg = 5.0
        self.skimming_cfg_positive = 5.0
        self.skimming_cfg_negative = 5.0

    sorting_priority = 14

    def title(self):
        return "Skimmed CFG for reForge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Adjust the settings for Skimmed CFG.</i></p>")

            enabled = gr.Checkbox(label="Enable Skimmed CFG", value=self.enabled)

            with gr.Group():
                mode = gr.Radio(
                    ["Single Scale", "Replace", "Linear Interpolation", "Dual Scales"],
                    label="Skimmed CFG Mode",
                    value=self.mode
                )

                with gr.Group() as single_scale_group:
                    skimming_cfg = gr.Slider(label="Skimming CFG", minimum=0.0, maximum=7.0, step=0.01, value=self.skimming_cfg)
                    full_skim_negative = gr.Checkbox(label="Full Skim negative", value=self.full_skim_negative)
                    disable_flipping_filter = gr.Checkbox(label="Disable Flipping Filter", value=self.full_skim_negative)

                with gr.Group() as lin_interp_group:
                    lin_interp_cfg = gr.Slider(label="Skimming CFG", minimum=0.0, maximum=10.0, step=0.01, value=self.lin_interp_cfg)

                with gr.Group() as dual_scales_group:
                    skimming_cfg_positive = gr.Slider(label="Skimming CFG Positive", minimum=0.0, maximum=10.0, step=0.01, value=self.skimming_cfg_positive)
                    skimming_cfg_negative = gr.Slider(label="Skimming CFG Negative", minimum=0.0, maximum=10.0, step=0.01, value=self.skimming_cfg_negative)

            def update_visibility(mode):
                return (
                    gr.Group.update(visible=(mode == "Single Scale")),
                    gr.Group.update(visible=(mode == "Linear Interpolation")),
                    gr.Group.update(visible=(mode == "Dual Scales"))
                )

            mode.change(
                update_visibility,
                inputs=[mode],
                outputs=[single_scale_group, lin_interp_group, dual_scales_group]
            )

            enabled.change(
                lambda x: self.update_enabled(x),
                inputs=[enabled]
            )

        return (enabled, mode, skimming_cfg, full_skim_negative, disable_flipping_filter, lin_interp_cfg, skimming_cfg_positive, skimming_cfg_negative)

    def update_enabled(self, value):
        self.enabled = value

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 8:
            self.enabled, self.mode, self.skimming_cfg, self.full_skim_negative, self.disable_flipping_filter, self.lin_interp_cfg, self.skimming_cfg_positive, self.skimming_cfg_negative = args[:8]
        else:
            logging.warning("Not enough arguments provided to process_before_every_sampling")
            return

        if not self.enabled:
            return

        unet = p.sd_model.forge_objects.unet.clone()

        if self.mode == "Single Scale":
            unet = CFG_skimming_single_scale_pre_cfg_node().patch(unet, self.skimming_cfg, self.full_skim_negative, self.disable_flipping_filter)[0]
        elif self.mode == "Replace":
            unet = skimReplacePreCFGNode().patch(unet)[0]
        elif self.mode == "Linear Interpolation":
            unet = SkimmedCFGLinInterpCFGPreCFGNode().patch(unet, self.lin_interp_cfg)[0]
        elif self.mode == "Dual Scales":
            unet = SkimmedCFGLinInterpDualScalesCFGPreCFGNode().patch(unet, self.skimming_cfg_positive, self.skimming_cfg_negative)[0]

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update({
            "skimmed_cfg_enabled": self.enabled,
            "skimmed_cfg_mode": self.mode,
            "skimming_cfg": self.skimming_cfg if self.mode == "Single Scale" else None,
            "full_skim_negative": self.full_skim_negative if self.mode == "Single Scale" else None,
            "disable_flipping_filter": self.disable_flipping_filter if self.mode == "Single Scale" else None,
            "lin_interp_cfg": self.lin_interp_cfg if self.mode == "Linear Interpolation" else None,
            "skimming_cfg_positive": self.skimming_cfg_positive if self.mode == "Dual Scales" else None,
            "skimming_cfg_negative": self.skimming_cfg_negative if self.mode == "Dual Scales" else None,
        })

        logging.debug(f"Skimmed CFG: Enabled: {self.enabled}, Mode: {self.mode}")

        return
