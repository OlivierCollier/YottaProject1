
from ..build_features import ClipTransformer

def test_clip_transformer():
	numbers = list(range(1, 10))
	transformer = ClipTransformer(0, 1)
	transformed_numbers = transformer.fit_transform(numbers)
	assert len(transformed_numbers) == len(numbers)



