import datasets
import huggingface_hub
import argparse
from PIL import Image, ImageOps
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage


class GenerateDataset:
    '''
    Generate Dataset Class.
    Uses korexyz/celeba-hq-256x256 dataset to convert to 128x128 and include a grayscale image.
    '''
    def __init__(self, source_repo, dest_repo, size, max_samples):
        '''Initializes dataset generation config.'''
        self.source = source_repo
        self.target = dest_repo
        self.res = size
        self.max_samples = max_samples

    def load_source(self):
        '''Loads source dataset from HuggingFace in streaming mode.'''
        self.source_data = datasets.load_dataset(self.source, streaming=True)

    def process_sample(self, sample: dict, idx: int) -> dict:
        '''Resizes image to target resolution and generates grayscale version.'''
        img = sample["image"]
        resized_img = img.resize((self.res, self.res), Image.LANCZOS)
        gray_img = ImageOps.grayscale(resized_img)
        return {"color": resized_img, "gray": gray_img, "source_id": str(idx)}

    def _make_split(self, split: str) -> list:
        '''Processes and returns samples for a given split as a list of dicts.'''
        if split == "train":
            source = self.source_data["train"]
        elif split == "validation":
            source = self.source_data["validation"].take(1000)
        else:
            source = self.source_data["validation"].skip(1000).take(1000)

        if self.max_samples:
            source = source.take(self.max_samples)

        samples = []
        for idx, sample in enumerate(source):
            samples.append(self.process_sample(sample, idx))
        return samples

    def build_dataset_dict(self) -> DatasetDict:
        '''Builds a DatasetDict with train, validation, and test splits.'''
        features = Features({
            "color": HFImage(),
            "gray": HFImage(),
            "source_id": Value("string")
        })

        train = Dataset.from_list(self._make_split("train"), features=features)
        val = Dataset.from_list(self._make_split("validation"), features=features)
        test = Dataset.from_list(self._make_split("test"), features=features)

        return DatasetDict({
            "train": train,
            "validation": val,
            "test": test
        })

    def push(self, dataset_dict: DatasetDict):
        '''Pushes the dataset to HuggingFace Hub.'''
        dataset_dict.push_to_hub(self.target, private=True)

    def run(self):
        '''Runs the full pipeline: load, build, push.'''
        self.load_source()
        dataset_dict = self.build_dataset_dict()
        self.push(dataset_dict)


def build_parser() -> argparse.ArgumentParser:
    '''Builds CLI argument parser for dataset generation script.'''
    parser = argparse.ArgumentParser(description="Dataset generation")
    parser.add_argument(
        "--source-repo",
        type=str,
        default=None,
        help="Hugging face dataset being used to generate the new dataset"
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=None,
        help="Repo to push dataset to using huggingface"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Target resolution for dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of samples per split for dry runs"
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    huggingface_hub.whoami()
    print("Dataset generation started")

    generator = GenerateDataset(
        source_repo=args.source_repo,
        dest_repo=args.output_repo,
        size=args.size,
        max_samples=args.max_samples
    )
    generator.run()


if __name__ == "__main__":
    main()
