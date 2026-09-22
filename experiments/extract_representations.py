"""Extract finding representations using a local Llama model."""
import argparse
from robust_medical_qa.io import read_json
from robust_medical_qa.runtime import add_runtime_arguments, load_runtime
from robust_medical_qa.extraction import extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('hidden', 'transmission'))
    parser.add_argument('--items', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prompt-config', default='configs/prompt.json')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--validate', type=int, default=0, help='Number of items for transmission intervention checks')
    add_runtime_arguments(parser)
    args = parser.parse_args()
    model, tok = load_runtime(args.model, args.tokenizer, args.device, args.dtype)
    extract(model, tok, args.items, args.output, read_json(args.config), read_json(args.prompt_config),
            args.kind, args.validate, args.limit)


if __name__ == '__main__':
    main()
