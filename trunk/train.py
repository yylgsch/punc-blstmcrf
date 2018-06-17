import bilsm_crf_model
import time

start = time.clock()
EPOCHS = 10


model, (train_x, train_y), (test_x, test_y) = bilsm_crf_model.create_model()
# train model
model.fit(train_x, train_y,batch_size=1000,epochs=EPOCHS, validation_data=[test_x, test_y])
model.save('model/crf.h5')
elapsed = (time.clock() - start)
print("总共花费时间（s）:", elapsed)