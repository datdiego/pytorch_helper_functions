'''
net_training_and_validation
  Input: Neural Network, Training data loader, Test/Validation data loader, Optimizer
  Output: 
    - Abstract: trained Neural Nework
    - Return: (tuple) trained Neural Network training loss, validation loss

'''
def net_training_and_validation(epochs, net, train_loader, validation_loader, optimizer, train_num):
  num_epochs = epochs
  # Establish a list for your loss history
  train_loss_history = list()
  validation_loss_history = list()

  for epoch in range(num_epochs):
      net.train()
      train_loss = 0.0
      train_correct = 0
      total = 0
      ## for each data batch 
      for i, data in enumerate(train_loader):
          # data is a list of [inputs, labels]
          inputs, labels = data

          # Pass to GPU if available.
          if torch.cuda.is_available():
              inputs, labels = inputs.cuda(), labels.cuda()

          optimizer.zero_grad()

          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          _, preds = torch.max(outputs.data, 1)
          total += labels.size(0)
          train_correct += (preds == labels).sum().item()
          train_loss += loss.item()
      print(f'EPOCH {epoch +1 }')
      print(f'\t Training accuracy: \t{ 100 * train_correct/total :.2f}% \t training loss: \t{train_loss/len(train_loader):.5f}')
      train_loss_history.append(train_loss/len(train_loader))


      validation_loss = 0.0
      validation_correct = 0
      total_val = 0
      net.eval()
      for inputs, labels in validation_loader:
          if torch.cuda.is_available():
              inputs, labels = inputs.cuda(), labels.cuda()

          outputs = net(inputs)
          loss = criterion(outputs, labels)

          _, preds = torch.max(outputs.data, 1)
          total_val += labels.size(0)
          validation_correct += (preds == labels).sum().item()
          validation_loss += loss.item()
      
      
      print(f'\t Validation accuracy: \t{100 * validation_correct/total_val:.2f}% \tvalidation loss: \t{validation_loss/len(validation_loader):.5f}')
      
      validation_loss_history.append(validation_loss/len(validation_loader))

 
  checkpoint = {'model': LeNet5(),
          'state_dict': net.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'epochs' : epochs,
          'dl_batch_size' : batch_size,
          'train loss' : train_loss/len(train_loader),
          'validation loss' : validation_loss/len(validation_loader)
          }

  torch.save(checkpoint, save_path +f'checkpoint_{date_num}_{train_num}_epoch_{(epoch+1)}.pth')
    

  return net, train_loss_history, validation_loss_history
