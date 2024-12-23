# marl with composite actions

this small example is meant to illustrate my struggles and attempts at adapting torchrl to multi agent ppo with composite action spaces.

my pain points were:
- individual log prob keys for the actions cause issues with stacking tensordicts somewhere internally
- non-natively multivariate distributions need special handling when calculating log probs
- ppo loss does not deal well with tensordicts, need to extract the sample log prob tensor
- petting zoo env wrapper does not properly split the action tensors when using dict action spaces