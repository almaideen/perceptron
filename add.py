from utils.model import perceptron

AND = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,0,0,1]
}

df = pd.DataFrame(AND)

X, y = prepare_data(df)

ETA = 0.3
EPOCHS = 10

model = perceptron(eta=ETA,epochs=EPOCHS)

model.fit(X,y)

_ = model.total_loss()