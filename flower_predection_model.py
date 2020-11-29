import pandas as pd

model = pd.read_pickle('IRIS_Model.bin')

sl = float(input('Enter Sepal_length(4.3 - 8.0) : '))
sw = float(input('Enter Sepal_width(2.0 - 4.4) : '))
pl = float(input('Enter Petal_length(1.0 - 7.0) : '))
pw = float(input('Enter Petal_width(0.1 - 2.5) : '))


result = model.predict([[pl, pw, sl*pw, sl*pl, sw*pl, sw*pw, pl*pw]])
if result == 1:
     result = 'Setosa'
elif result == 0:
     result = 'Virginica'
elif result == 2:
     result = 'Versicolor'
print('According to Your information this flower belongs to {} species'.format(result))
#print(result)
