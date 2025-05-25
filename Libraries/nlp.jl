// nlp.joyl - Professional NLP toolkit
pub struct Tokenizer {
    vocab: {string: int},
    special_tokens: {string: int},
    max_length: int
}

impl Tokenizer {
    /// Load pretrained tokenizer
    pub fn from_pretrained(name: string) -> Tokenizer {
        let path = download_model(name);
        let config = parse_config(path + "/config.json");
        
        Tokenizer {
            vocab: load_vocab(path + "/vocab.txt"),
            special_tokens: config.special_tokens,
            max_length: config.max_length
        }
    }

    /// Tokenize text with full preprocessing
    pub fn tokenize(&self, text: string) -> [int] {
        let cleaned = text
            .to_lowercase()
            .replace(r"[^\w\s]", "")
            .split_whitespace()
            .collect();
        
        let mut tokens = [self.special_tokens["[CLS]"]];
        
        for word in cleaned {
            if self.vocab.contains_key(word) {
                tokens.push(self.vocab[word]);
            } else {
                tokens.push(self.special_tokens["[UNK]"]);
            }
        }
        
        tokens.truncate(self.max_length - 1);
        tokens.push(self.special_tokens["[SEP]"]);
        
        return tokens;
    }
}

pub struct Transformer {
    model: NeuralNetwork,
    tokenizer: Tokenizer
}

impl Transformer {
    /// Initialize transformer model
    pub fn new(model_name: string) -> Transformer {
        Transformer {
            model: load_pretrained_model(model_name),
            tokenizer: Tokenizer::from_pretrained(model_name)
        }
    }

    /// Generate embeddings for text
    pub fn embed(&self, text: string) -> Tensor {
        let tokens = self.tokenizer.tokenize(text);
        let input = Tensor::from_array(tokens).reshape([1, -1]);
        self.model.forward(input)
    }
}