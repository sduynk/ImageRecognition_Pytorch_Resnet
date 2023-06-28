from torch import optim
from tqdm import tqdm
import torch
import os

def train(model,
          optimizer,
          loss_fn,
          train_loader,
          valid_loader,
          epochs=50,
          device='cpu',
          verbose=True,
          path=r'C:\Users\sedau\Desktop\AIC Group\ML\HSP_ML_seda\weights',
          fname=None,
          update=2):
    
    
    #sz=len(train_loader)
    #basename=get_base_fname()# base file name based on time stamps

    model=model.to(device)
    
    test_acc_list=[]
    test_loss_list=[]
    training_acc_list=[]
    training_loss_list=[]

    for epoch in tqdm(range(epochs)):
        #start=time()
        epoch_train_loss =0 
        epoch_test_loss =0 
        correct_train = 0
        correct_test = 0

        model.train()
        for idx,(image_lr,labels) in enumerate(train_loader):

            image_lr=image_lr.to(torch.float32).to(device)
            #labels = labels.reshape(-1,1).to(device)
            labels = labels.to(device)

            # train discriminator
            optimizer.zero_grad()
            output = model(image_lr).flatten()
            
            #print(output)
            #print(labels)

            loss=loss_fn(output,labels)
            loss.backward(retain_graph=True)

            optimizer.step()

            correct_train += count_correct_labels(output,labels)
            epoch_train_loss += loss.item()

        training_acc=correct_train/len(train_loader.sampler)
        training_acc_list.append(training_acc)
        training_loss_list.append(epoch_train_loss)
        
        model.eval()
        for idx,(image_lr,labels) in enumerate(valid_loader):

            image_lr=image_lr.to(torch.float32).to(device)
            #labels = labels.reshape(-1,1).to(device)
            labels = labels.to(device)

            output = model(image_lr).flatten()
            loss=loss_fn(output,labels)
            correct_test += count_correct_labels(output,labels)
            epoch_test_loss += loss.item()

        test_acc=correct_test/len(valid_loader.sampler)
        test_acc_list.append(test_acc)
        test_loss_list.append(epoch_test_loss)

        print('epoch: {},train loss: {}, valid loss: {}, train acc: {}, valid acc: {}'.format(epoch,epoch_train_loss,epoch_test_loss,training_acc,test_acc))

        if (epoch+1)%update == 0:
            if fname:
                #fname2 =fname+'weights_'+str(epoch)+'_lr_'+str(lr)+'.pt'
                fname2 =fname+'weights_'+str(epoch)+'.pt'
            else:
                #fname2 ='weights_'+str(epoch)+'_lr_'+str(lr)+'.pt'
                fname2 ='weights_'+str(epoch)+'.pt'
            
            torch.save(model.state_dict(), os.path.join(path,fname2))
    return evals(training_loss_list,test_loss_list,training_acc_list,test_acc_list)

def count_correct_labels(predictions,targets):
    return ((predictions>0.5)==targets).cpu().sum().item()#(targets==predictions.flatten()).sum().item()



class evals:
    def __init__(self,train_loss,test_loss,train_acc,test_acc):
        self.train_loss=train_loss
        self.test_loss=test_loss
        self.train_acc=train_acc
        self.test_acc=test_acc
