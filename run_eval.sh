#!/bin/bash
mkdir -p eval








# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t0-3b-3b_stac_focus.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_focus.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_focus.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_focus.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5-3b_stac_focus.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5-large_stac_focus.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_focus.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_focus.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_focus.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t0-3b-3b_stac_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_natural2.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5-3b_stac_natural2.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5-large_stac_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_natural2.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t0-3b-3b_stac_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_augmented.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5-3b_stac_augmented.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5-large_stac_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_augmented.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t0-3b-3b_stac_natural.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_natural.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_natural.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_natural.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5-3b_stac_natural.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5-large_stac_natural.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_natural.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_natural.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_natural.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t0-3b-3b_molweni_focus.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_focus.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_focus.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_focus.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5-3b_molweni_focus.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5-large_molweni_focus.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_focus.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_focus.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_focus.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t0-3b-3b_molweni_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_natural2.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_natural2.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5-3b_molweni_natural2.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5-large_molweni_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_natural2.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_natural2.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t0-3b-3b_molweni_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_augmented.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_augmented.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5-3b_molweni_augmented.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5-large_molweni_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_augmented.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_augmented.txt
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t0-3b-3b_molweni_natural.txt
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_natural.txt
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_natural.txt
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_natural.txt
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5-3b_molweni_natural.txt
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5-large_molweni_natural.txt
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_natural.txt
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_natural.txt
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_natural.txt