import argparse

def validate_args(parser, args):
	if len(args.translation_languages) > 3 or len(set(args.translation_languages)) < len(args.translation_languages):
		parser.error("Invalid number of languages (> 3) or duplicate language code given.")

def parse_arguments():
	parser = argparse.ArgumentParser(description="This script is responsive for running the whole experiment pipeline for this IR experiment.")
	parser.add_argument('--translation-target', nargs=1, choices=["queries", "docs"], default=["queries"],
							help='(Optional - Preprocessing) Specifies the translation target - either queries or docs. If not given the documents are translated, which means that in the experiment a english query is matched to multilingual translated documents.')
	parser.add_argument('--translation-languages', nargs="+", choices=["de", "it", "fr"], default=["de", "it", "fr"],
							help='(Optional - Preprocessing) Specifies the languages for the translation of queries or documents. If not given the targets are translated into all three languages.')

	args = parser.parse_args()

	validate_args(parser, args)

	return args