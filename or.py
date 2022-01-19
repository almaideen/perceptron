OR = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,1,1,1]
}

df = pd.DataFrame(OR)

X, y = prepare_data(df)

ETA = 0.3
EPOCHS = 10

OR_model = perceptron(eta=ETA,epochs=EPOCHS)

OR_model.fit(X,y)

_ = OR_model.total_loss()