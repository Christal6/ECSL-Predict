#!/usr/bin/env python3
import sys
from Bio import GenBank
from Bio import SeqIO

def seq_get():
	gbk_filename = "data/GCF_000005845.2_ASM584v2_genomic.gbff"
	faa_filename = "data/gbff_converted.faa"

	input_handle = open(gbk_filename, "r")
	output_handle = open(faa_filename, "w")

	for seq_record in SeqIO.parse(input_handle, "genbank") :
		print("Dealing with GenBank record %s" % seq_record.id)
		for seq_feature in seq_record.features:
			if seq_feature.type == "CDS":
				print(seq_feature)
				if 'translation' in seq_feature.qualifiers.keys():
					# output_handle.write(">%s, from %s protein_id=%s product=%s\n%s\n" % (
					output_handle.write(">%s\n%s\n" % (
						seq_feature.qualifiers['locus_tag'][0],
						# seq_record.name,
						# seq_feature.qualifiers['protein_id'][0],
						# seq_feature.qualifiers['product'][0],
						seq_feature.qualifiers['translation'][0]))
	output_handle.close()
	input_handle.close()
	print("Done")


def blast(genome1,genome2):
	# makeblastdb
	cmd = f"makeblastdb -in blastdatabase/{genome2} -dbtype prot -out blastdatabase/{genome2}"
	os.system(cmd)

	#-query genome1
	cmd = f"blastp -num_threads 48 -db blastdatabase/{genome2} -query protFile/{genome1} -outfmt 6 -evalue 1e-5 -num_alignments 5 -out blastp_{genome1}_{genome2}.txt"
	os.system(cmd)


if __name__ == "__main__":
	seq_get()