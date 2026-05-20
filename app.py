import pandas as pd
import joblib
import gradio as gr

model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptoms = joblib.load("symptoms.pkl")

symptoms = [s.strip().lower() for s in symptoms]

def predict_disease(selected_symptoms):

    selected_symptoms = [s.strip().lower() for s in selected_symptoms]

    user_dict = {symptom: 0 for symptom in symptoms}

    for symptom in selected_symptoms:
        if symptom in user_dict:
            user_dict[symptom] = 1

    input_df = pd.DataFrame([user_dict], columns=symptoms)

    pred = model.predict(input_df)[0]

    return le.inverse_transform([pred])[0]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Disease Prediction Web App")
    gr.Markdown("Select your symptoms from either the search dropdown or the checkbox group below to analyze potential conditions.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Search Symptoms")
            dropdown = gr.Dropdown(
                choices=symptoms,
                multiselect=True,
                label="Type to search symptoms..."
            )

            gr.Markdown("### Quick-Select Symptoms")
            checkbox = gr.CheckboxGroup(
                choices=symptoms,
                label="Check all that apply"
            )
            
            btn = gr.Button("Predict Condition", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Diagnostic Output")
            output = gr.Textbox(label="Predicted Diagnosis", interactive=False)
    def combined_predict(d, c):
        selected = list(set((d or []) + (c or [])))
        if not selected:
            return "Please select or search for at least one symptom!"
        return predict_disease(selected)

    btn.click(combined_predict, inputs=[dropdown, checkbox], outputs=output)
if __name__ == "__main__":
    demo.launch()