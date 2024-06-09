import tensorflow as tf

# Define the path to the saved model and the specific checkpoint
model_path = 'C:/Users/Eichleitner/Documents/Coding/model.h5'
checkpoint_path = 'C:/Users/Eichleitner/Documents/Coding/training_checkpoints/cp-0007.ckpt'

# Load the entire model, including its architecture and weights
model = tf.keras.models.load_model(model_path)

# Load the weights from checkpoint 12 into the model
try:
    # TensorFlow 2.x expects the checkpoint without the '.index' or '.data-00000-of-00001' extensions
    # Only provide the prefix up to the checkpoint number
    model.load_weights(checkpoint_path)

    print(f"Checkpoint '{checkpoint_path}' loaded successfully.")

    # Save the updated model back to 'model.h5'
    model.save(model_path)
    print(f"Model saved back to '{model_path}'.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
