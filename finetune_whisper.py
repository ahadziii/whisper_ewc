c

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Outputs logs to stdout
    ]
)

logger = logging.getLogger(__name__)

#######################     ARGUMENT PARSING        #########################

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
# Paths for previous models and their data
parser.add_argument(
    '--prev_models', 
    nargs='+',
    type=str, 
    required=True,
    help='List of paths to the previous models.')

parser.add_argument(
    '--fisher_audio_paths_files', 
    type=str, 
    required=True,
    help='List of files containing Fisher audio file paths for each previous model.')

parser.add_argument(
    '--fisher_transcriptions_files', 
    type=str, 
    required=True,
    help='List of files containing Fisher transcriptions for each previous model.')

parser.add_argument(
    '--lambda_ewc', 
    type=float, 
    default=0.1,
    help='EWC regularization strength.')

parser.add_argument(
    '--fisher_samples',
    type=int,
    default=100,
    help='Number of samples to compute Fisher Information.')

parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--device', 
    type=str, 
    default='cuda',
    help='Device to use for training (e.g., "cuda" or "cpu").')

parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=1, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=2000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=1, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default=[], 
    help='List of datasets to be used for evaluation.'
)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

logger.info('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
logger.info('ARGUMENTS OF INTEREST:')
logger.info(vars(args))
logger.info('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = False
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = True
do_remove_punctuation = True
normalizer = BasicTextNormalizer()
scaler = torch.GradScaler("cuda")

#############################     EWC IMPLEMENTATION    #####################################
#preprocess audio for EWC
def preprocess_audio(file_path, processor, device):
    """
    Load and preprocess an audio file with error handling.

    Args:
        file_path (str): Path to the audio file.
        processor (WhisperProcessor): Processor for the Whisper model.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Preprocessed input features.
    """
    try:
        # Load audio file
        speech_array, sampling_rate = torchaudio.load(file_path)
        speech_array = speech_array.squeeze() 

        # Resample if the audio is not at 16kHz
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech_array = resampler(speech_array)

        # Convert speech array to numpy and process
        input_features = processor(speech_array.numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device)
        return input_features
    except Exception as e:
        logger.info(f"Error preprocessing audio {file_path}: {e}")
        return None
    
def load_data(audio_paths_file, transcriptions_file):
    """
    Load audio file paths and corresponding transcriptions from text files with error handling.

    Args:
        audio_paths_file (str): Path to the file containing audio file paths.
        transcriptions_file (str): Path to the file containing transcriptions.

    Returns:
        list of tuples: Each tuple contains (audio_file_path, transcription)
    """
    audio_paths = []
    transcriptions = []
    try:
        with open(audio_paths_file, "r", encoding='utf-8') as f_audio, open(transcriptions_file, "r", encoding='utf-8') as f_text:
            audio_paths = [line.strip() for line in f_audio if os.path.isfile(line.strip())]
            transcriptions = [line.strip() for line in f_text]
    except Exception as e:
        logger.info(f"Error loading data: {e}")
    # Ensure both lists are of the same length
    min_length = min(len(audio_paths), len(transcriptions))
    if len(audio_paths) != len(transcriptions):
        logger.info(f"Warning: Number of audio paths ({len(audio_paths)}) and transcriptions ({len(transcriptions)}) do not match. Truncating to {min_length} samples.")
    return list(zip(audio_paths[:min_length], transcriptions[:min_length]))



# Fisher matrix computation
def compute_fisher_information(model, processor, dataset, device="cuda", num_samples=11592):

    model.to(device)
    model.train()
    fisher_information = defaultdict(float)
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)

    for idx, (file_path, transcription) in enumerate(dataset[:num_samples]):
        input_features = preprocess_audio(file_path, processor, device)
        if input_features is None:
            continue  # Skip if preprocessing failed

        # Encode the transcription into labels
        labels = processor.tokenizer(transcription, return_tensors="pt", padding=True).input_ids.to(device)

        # Compute loss
        outputs = model(input_features, labels=labels)
        loss = outputs.loss
        model.zero_grad()
        loss.backward()

        # Accumulate Fisher Information
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_information[name] += (param.grad.detach() ** 2) / num_samples
                # fisher_information[name] += (param.grad.data ** 2) / num_samples  # Average over samples

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx + 1}/{num_samples} samples for Fisher Information")

    return fisher_information

###############################    EWC Loss Implementation    ###########################

class EWCLossSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, fisher_matrix=None, previous_params=None, lambda_ewc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_matrix = fisher_matrix
        self.previous_params = previous_params
        self.lambda_ewc = lambda_ewc

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure num_items_in_batch is passed correctly
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        ewc_loss = 0.0

        for name, param in model.named_parameters():
            # Remove 'module.' prefix if present
            if name.startswith('module.'):
                adjusted_name = name[len('module.'):]
            else:
                adjusted_name = name

            
            # logger.info(f"Adjusted parameter name: {adjusted_name}")

            if adjusted_name in self.fisher_matrix:
                fisher = self.fisher_matrix[adjusted_name].to(param.device)
                prev_param = self.previous_params[adjusted_name].to(param.device)
                ewc_term = (fisher * (param - prev_param).pow(2)).sum()
                ewc_loss += ewc_term
                # logger.info(f"EWC loss for {adjusted_name}: {ewc_term.item()}")
            else:
                logger.info(f"Parameter {adjusted_name} not found in Fisher matrix.")

        # if self.lambda_ewc is not None:
        #     loss += (self.lambda_ewc / 2) * ewc_loss
        #     logger.info(f"Total EWC loss added: {(self.lambda_ewc / 2) * ewc_loss} and loss is {loss}")

        if self.lambda_ewc is not None:
            loss += (self.lambda_ewc) * ewc_loss
            logger.info(f"Total EWC loss added: {(self.lambda_ewc) * ewc_loss} and loss is {loss}")

        return (loss, outputs) if return_outputs else loss



###############################    FIM Calculation      ###########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained(args.model_name, language="English", task="transcribe")
param_model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
previous_params = {name: param.data.to(device) for name, param in param_model.named_parameters()}

# Compute Fisher Information for previous models

if args.prev_models and args.fisher_audio_paths_files and args.fisher_transcriptions_files:
    prev_models = []
    
    # Load the dataset
    logger.info("Loading dataset for Fisher Information calculation...")
    fisher_dataset = load_data(args.fisher_audio_paths_files, args.fisher_transcriptions_files)
    
    # Load the models
    for model_path in args.prev_models:
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        prev_models.append(model)
    
    # Compute and accumulate FIMs for all models
    accumulated_fisher_information = defaultdict(float)
    logger.info("Computing Fisher Information for previous models...")
    for model in prev_models:
        fisher_information = compute_fisher_information(
            model=model,
            processor=processor,
            dataset=fisher_dataset,
            device=device,
            num_samples=args.fisher_samples
        )
        # Accumulate the FIMs
        for name in fisher_information:
            accumulated_fisher_information[name] += fisher_information[name]
    
    logger.info("Accumulated Fisher Information computed.")

    # Move Fisher Information to device
    for key in accumulated_fisher_information:
        accumulated_fisher_information[key] = accumulated_fisher_information[key].to(device)

    # Move previous parameters to device
    for key in previous_params:
        previous_params[key] = previous_params[key].to(device)

# After computing Fisher Information and previous parameters
fisher_matrix = {name: fisher.detach().clone() for name, fisher in accumulated_fisher_information.items()}
previous_params = {name: param.detach().clone() for name, param in previous_params.items()}




# logger.info("Keys in Fisher matrix:")
# logger.info(list(fisher_matrix.keys()))
    
 


#############################       MODEL LOADING       #####################################



feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
model.config.use_cache = True


if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

if gradient_checkpointing:
    model.config.use_cache = False


############################        DATASET LOADING AND PREP        ##########################

def load_custom_dataset(split):
    ds = []
    if split == 'train':
        for dset in args.train_datasets:
            ds.append(load_from_disk(dset))
    if split == 'eval':
        for dset in args.eval_datasets:
            ds.append(load_from_disk(dset))

    ds_to_return = concatenate_datasets(ds)
    ds_to_return = ds_to_return.shuffle(seed=22)
    return ds_to_return

def prepare_dataset(batch):
    # Load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # Compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # Compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # Optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    
    # Remove unnecessary columns
    return {
        "input_features": batch["input_features"],
        "labels": batch["labels"],
    }

max_label_length = model.generation_config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length



logger.info('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_custom_dataset('train')
raw_dataset["eval"] = load_custom_dataset('eval')

raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=args.sampling_rate))
raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
)

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Prepare inputs
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Prepare labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step, remove it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
logger.info('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode the predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Apply normalization if required
    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]


    # Filter out empty predictions and labels
    filtered_pred_str = []
    filtered_label_str = []

    for p, l in zip(pred_str, label_str):
        if p.strip() and l.strip():
            filtered_pred_str.append(p)
            filtered_label_str.append(l)

    # Calculate WER only for non-empty predictions and labels
    wer = 100 * metric.compute(predictions=filtered_pred_str, references=filtered_label_str)
    return {"wer": wer}


###############################     TRAINING ARGS AND TRAINING      ###########################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
        push_to_hub=False,
        remove_unused_columns=False,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
        push_to_hub=False,
        
    )

trainer = EWCLossSeq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    fisher_matrix=fisher_matrix,
    previous_params=previous_params,
    lambda_ewc=args.lambda_ewc,
)

processor.save_pretrained(training_args.output_dir)

logger.info('TRAINING IN PROGRESS...')
trainer.train()
logger.info('DONE TRAINING')