"""Validator PvP prompt alignment package.

Houses the canonical pvp_game_prompts.yml (byte-copied from
G.O.D/core/config/pvp_game_prompts.yml) and the loader functions that
trajectory generators import to produce the exact same system prompt
the validator's PvP eval container uses.

Training-time prompt mismatch was the single largest gap behind the
catastrophic LD 0-200 eval result: model trained on a hand-written
prompt with extra "Strategy Tips" sections lost every game when the
validator handed it the YAML-template prompt at inference time. This
package closes that gap.
"""
