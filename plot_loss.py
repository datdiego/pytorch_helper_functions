import matplotlib

'''
Return plot of training loss history and validation loss history
'''
# Plot the training and validation loss history
def plot_loss(train_loss_history, validation_loss_history):
  plt.plot(train_loss_history, label="Training Loss")
  plt.plot(validation_loss_history, label="Validation Loss")
  plt.grid(color='lightblue')
  plt.legend()
  plt.show()
