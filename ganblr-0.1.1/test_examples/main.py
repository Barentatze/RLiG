from ganblr import get_demo_data
from ganblr.models import GANBLR

# this is a discrete version of adult since GANBLR requires discrete data.
df = get_demo_data('adult')
x, y = df.values[:,:-1], df.values[:,-1]

model = GANBLR()
model.fit(x, y, epochs = 10)

#generate synthetic data
synthetic_data = model.sample(1000)
