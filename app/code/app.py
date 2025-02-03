from flask import Flask, render_template, request, jsonify
import pickle
import torch
from utils import Encoder, Decoder, Seq2SeqTransformer, initialize_weights, fetch_text_transform, my_tokenizer

app = Flask(__name__)

# Loading the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtt_additive_data = pickle.load(open("models/mtt_additive.pkl", "rb"))

# Define Transformers
# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'my'

token_transform = mtt_additive_data['token_transform']
vocab_transform = mtt_additive_data['vocab_transform']
text_transform = fetch_text_transform(token_transform, vocab_transform)
# print("token_transform ", token_transform)
# print("vocab_transform ", vocab_transform)
# print('text_transform ', text_transform)

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TRG_LANGUAGE])
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM,
              HID_DIM,
              ENC_LAYERS,
              ENC_HEADS,
              ENC_PF_DIM,
              ENC_DROPOUT,
              device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

SRC_PAD_IDX = PAD_IDX
TRG_PAD_IDX = PAD_IDX

model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.apply(initialize_weights)

# Load the pre-trained model weights
# model.load_state_dict(torch.load("models/Seq2SeqTransformer_additive.pt", map_location=device))
# model.eval()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    model.load_state_dict(torch.load("models/Seq2SeqTransformer_additive.pt", map_location=device))

    data = request.get_json()
    user_prompt = data.get("prompt")
    if not user_prompt:
        return jsonify({"error": "Please provide a prompt."}), 400

    user_prompt = user_prompt.strip()
    # print("user prompt: ", user_prompt)
    try:
        # Preprocess the input text
        src_text = text_transform[SRC_LANGUAGE](user_prompt).to(device)
        # print("length of input src text: ", len(src_text))
        src_text = src_text.reshape(1, -1)
        src_mask = model.make_src_mask(src_text)
        # print("src mask length: ", len(src_mask))

        max_seq = 100
        model.eval()
        with torch.no_grad():
            enc_output = model.encoder(src_text, src_mask)

        # Generate translation
        outputs = []
        input_tokens = [EOS_IDX]
        for i in range(max_seq):
            with torch.no_grad():
                starting_token = torch.LongTensor(input_tokens).unsqueeze(0).to(device)
                trg_mask = model.make_trg_mask(starting_token)
                output, _ = model.decoder(starting_token, enc_output, trg_mask, src_mask)

            pred_token = output.argmax(2)[:, -1].item()
            input_tokens.append(pred_token)
            outputs.append(pred_token)

            if pred_token == EOS_IDX:
                break

        # Convert tokens to text
        trg_tokens = [vocab_transform[TRG_LANGUAGE].get_itos()[i] for i in outputs]
        # Remove special symbols from the output
        trg_tokens = [token for token in trg_tokens if token not in special_symbols]
        translated_text = " ".join(trg_tokens[1:-1])

        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": f"Error during translation: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
