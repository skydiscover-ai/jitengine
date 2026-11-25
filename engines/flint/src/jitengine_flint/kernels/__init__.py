"""The decode megakernel: one persistent Metal dispatch per pass over all 49 layers and the head (metal_src, with
vmoe_src for the verify pass's MoE block, and kv8_src with kv8_fold for the kv8 profile), its plan (layouts, the
attention grid, descriptors, parameters) and its runner (compile, bind, encode, probe)."""
