# DFlash/Qwen3.6 block_size=832 origin hypothesis check

## Checked explicit knobs

Visible service/config knobs:

- `docker-compose.yml`: `num_speculative_tokens=${VLLM_DFLASH_N:-8}`
- DFlash local `config.json`:
  - `architectures=["DFlashDraftModel"]`
  - `hidden_size=5120`
  - `num_hidden_layers=5`
  - `num_attention_heads=32`
  - `num_key_value_heads=8`
  - `head_dim=128`
  - `block_size=16`
  - `sliding_window=2048`
  - `dflash_config.target_layer_ids=[1,16,31,46,61]`
  - `layer_types=[sliding_attention, sliding_attention, sliding_attention, sliding_attention, full_attention]`

Target local `text_config`:

- `architectures=["Qwen3_5ForConditionalGeneration"]`
- `hidden_size=5120`
- `num_attention_heads=24`
- `num_key_value_heads=4`
- `head_dim=256`
- `layer_types` alternates 3 `linear_attention` layers and 1 `full_attention` layer.

No explicit config was found for a draft cache chunk length or target feature projection token length that directly explains `832`.

## Hypothesis 1: hidden features become virtual tokens

Assessment: unlikely from current code.

DFlash does inject target hidden states, but the implementation projects target/context hidden states into K/V using the real `context_states.shape[0]` and writes them using `context_slot_mapping`. The feature fusion path uses `fc(hidden_states)` with `target_layer_ids`, but this changes feature width, not cache token count. No code path observed that turns the 5 selected target layer features into extra virtual KV tokens.

## Hypothesis 2: hybrid cache geometry alignment

Assessment: correct direction, but the exact reason is page-byte alignment with target hybrid linear-attention/Mamba-style state, not simply head-size LCM.

The target model is the hybrid one: `head_size=256`, `gqa=6`, linear/full attention layer mix. DFlash drafter is `head_size=128`, `gqa=4`, `block_size=16` in its own config.

vLLM computes the hybrid block size in `vllm/platforms/interface.py`:

```text
attn_block_size = alignment * ceil(
    mamba_page_size / (alignment * attn_page_size_1_token)
)
```

For the observed service, this yields `832`. The fact that `832 = 13 * 64` is incidental; the important relation is that it is the next multiple of the backend alignment above about `823` attention-token-equivalent bytes of Mamba/linear-attention state.

## Hypothesis 3: DFlash block diffusion fixed window

Assessment: unlikely as the root cause of `832`.

The service has `num_speculative_tokens=8`, but the block-size alignment path is a startup-time cache geometry calculation based on model/cache specs. It does not read `num_speculative_tokens` or DFlash diffusion window settings. DFlash may affect runtime attention calls and cache traffic, but the `832` value is produced before request-time speculative decode behavior.

## Padding to 1024 check

Padding `832` to `1024` does not activate native ROCm paged attention on gfx1x.

Direct predicate check:

- `head_size=128, block_size=16, gqa=4` => native allowed
- `head_size=128, block_size=32/64/832/1024` => native not allowed
- `head_size=256, block_size=16/32/64/832/1024` => native not allowed

The current gfx1x predicate is not "any power of two". It is effectively `head_size == 128` and `block_size == 16`. The native path also has `_PARTITION_SIZE_ROCM=256` and requires `256 % block_size == 0`; `1024` violates that partition assumption.

## Practical conclusion

The best explanation for `832` is target-model hybrid cache geometry. DFlash config does expose feature-selection knobs (`target_layer_ids`) and speculative token count, but not a length knob that directly controls `832`.

The promising framework-level route is not padding to `1024`; it is decoupling attention page/block size from target hybrid Mamba/linear-attention page size so the DFlash drafter `head_size=128/gqa=4` path can potentially use native `block_size=16`. The target `head_size=256/gqa=6` path still needs Triton or a new ROCm kernel.
