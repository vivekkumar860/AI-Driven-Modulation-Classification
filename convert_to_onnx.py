import tf2onnx

output_path = "mod_classifier.onnx"
model_proto, _ = tf2onnx.convert.from_saved_model("saved_model", output_path=output_path)
print(f"Model converted and saved as {output_path}") 