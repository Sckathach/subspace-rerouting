## Research

- evolution of logits diff with head ablations
- plot optimization path with same seed: clip the nearest possible token at each step/ clip every n step/ no clip until the end
- cutting one head = cutting n heads?
- test perplexity -> qwen sucks -> because duration too low? -> if ppl works well -> several backprops until embed, then move to tokens with loss + ppl OR ppl in loss
- which params are the best for each models, are there obvious correlations? 
- which types of attacks work/don't work (bomb hard, cyber easy) + relationship with the training dataset?
- transfer models (bad)/ system message (seems OK)
- modify ssr to return best str (the suffix) every m_steps OR early stop with verification -> see the evolution of the suffix through the steps: could we stop earlier? Or do we need a full run (~250) -> spoiler "no" (I think)
- are steering and probe attacks similar? (i.e.: is the obtained suffix similar (in distribution or just in structure (i.e.: do both attacks find suffixes that work to trigger the "Disclaimer, this is for education only")))
- find patterns between models, are the prompts where gemma resists also those where llama resists? (weird)
- what about "resisting"? Is it proportional to the time spent optimizing? Is it related to the final loss? Is it related to a rapid drop in loss (i.e.: the loss quickly drops to something low, then stagnates)? => in this case we could put verifiers in SSR to change optimization parameters in real time (i.e.: the prompt passed in S1, I can modify the loss to target S2) (I think yes, the success seems related to the first loss drops & final loss)
- get the distributions of attacks and compare with gcg (onehot/num alive/ variance/ final tokens in initial dist?)
- PCA diff attacks
- dist heads with diff attacks
- dist shifts with SAE
- SAE concept level features -> look for consistency
- improve attribution with SAE (DLA, ablation, patching)
- (SAE) safety heads -> circuits?
- SAE steering <https://arxiv.org/pdf/2501.17148 https://arxiv.org/pdf/2411.11296>
- SAE interp of latents by backprop on input nn
- step n times and see if consistent flips -> distribution changes => control gen with ppl

## Code improvement

- create the distribution of the best adversarial tokens then initialize with that (random uniform over the distribution of the best ones)
- use `<unk>` instead of x
- optimize in embedding space only (first tests showed that may not work)
- intelligent buffer extraction i.e.: not candidate+archive but sorted(candidate+archive)

