# Local_Remote
**Description: This tutorial shows how to send data and parse results between the local model (clone) and the remote model (victim).**

## 👊 Example
⚠ **Victim's response_type should be set to hard-label or soft-label.** ⚠
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
batch_size = 8
image_size = 224
batch_image = torch.randn([batch_size, 3, image_size, image_size])
# Normalize batch_image to [0, 1], equal to transforms.ToTensor().
batch_image = (batch_image - batch_image.min()) / (batch_image.max() - batch_image.min())
# Initialize remote victim model.
victim = VictimCaller(
    device=device,
    victim_model_name='r18.military-20',
    response_type='xxx-label',
    remote_api='http://127.0.0.1:5500/api/victim-classifier',
)
```
### response_type='hard-label'
```python
print(result)
# ['r18.military-20', 'hard-label', tensor([18, 14, 18, 18, 18, 18, 14, 14], device='cuda:0')]
print(type(result))
# <class 'str'>
target_part = re.search(r'\[(\d+\s*,?\s*)+\]', result).group(0)
target_part = [int(num) for num in target_part.strip('[]').split(',')]
target_part = torch.tensor(target_part)
print(target_part)
# tensor([18, 14, 18, 18, 18, 18, 14, 14])
print(type(target_part))
# <class 'torch.Tensor'>
```
### response_type='soft-label'
```python
print(result)
# ['r18.military-20', 'soft-label', tensor([[-1.4863, -1.0747, 1.3176, -1.2749, -0.7421, -0.3141, -3.1253,
# -3.8630, 0.0107, -0.7535, 0.3905, 0.3062, -0.5542, -0.1532, 3.8926, 3.0952, -0.5869, -0.3717, 4.0759,
# 3.6533]], device='cuda:0', grad_fn=<AddmmBackward0>)]
print(type(result))
# <class 'str'>
target_part = re.search(r'(\[\[.*?\]\])', result, re.DOTALL).group(0)
target_part = ast.literal_eval(target_part)
target_part = torch.tensor(target_part)
print(type(target_part), target_part.size())
 # <class 'torch.Tensor'> torch.Size([2, 20])
```

## ☝ From local to remote

## ✌ From remote to local
