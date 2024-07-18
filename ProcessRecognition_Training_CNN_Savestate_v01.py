import tensorflow as tf
import re

model_used = "model_process_step_recognition.keras"

# Define the path to the saved model and the specific checkpoint
model_path = "C:/Users/LangeMatteoG/Documents/ProcessStepRecogntionRST/" + model_used
checkpoint_path = "C:/Users/LangeMatteoG/Documents/ProcessStepRecogntionRST/training_checkpoints/cp-0001.ckpt.keras"

# Load the entire model, including its architecture and weights
model = tf.keras.models.load_model(model_path)

# Load the weights from checkpoint 12 into the model
try:
    # Only provide the prefix up to the checkpoint number
    model.load_weights(checkpoint_path)

    print(f"Checkpoint '{checkpoint_path}' loaded successfully.")

    model.save(model_path)
    print(f"Model saved back to '{model_path}'.")

except Exception as e:
    print(f"Error loading checkpoint: {e}")
