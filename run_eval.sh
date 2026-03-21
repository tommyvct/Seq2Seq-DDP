#!/bin/bash
mkdir -p eval








# echo "Evaluating t0-3b-3b on stac with focus structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s focus --lr 2e-5 --seed 27 > eval/t0-3b-3b_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on stac with focus structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on stac with focus structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on stac with focus structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on stac with focus structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5-3b_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on stac with focus structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5-large_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on stac with focus structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on stac with focus structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on stac with focus structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on stac with natural2 structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s natural2 --lr 2e-5 --seed 27 > eval/t0-3b-3b_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on stac with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on stac with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on stac with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on stac with natural2 structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5-3b_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on stac with natural2 structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5-large_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on stac with natural2 structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on stac with natural2 structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on stac with natural2 structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on stac with augmented structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s augmented --lr 2e-5 --seed 27 > eval/t0-3b-3b_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on stac with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on stac with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on stac with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on stac with augmented structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5-3b_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on stac with augmented structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5-large_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on stac with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on stac with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on stac with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on stac with natural structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus stac --test_corpus stac -s natural --lr 2e-5 --seed 27 > eval/t0-3b-3b_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on stac with natural structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-base_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on stac with natural structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-large_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on stac with natural structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/flan-t5-xl_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on stac with natural structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5-3b_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on stac with natural structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5-large_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on stac with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-270m_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on stac with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-1b_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on stac with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus stac --test_corpus stac -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-4b_stac_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on molweni with focus structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s focus --lr 2e-5 --seed 27 > eval/t0-3b-3b_molweni_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on molweni with focus structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on molweni with focus structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_focus.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on molweni with focus structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on molweni with focus structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5-3b_molweni_focus.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on molweni with focus structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5-large_molweni_focus.txt
# echo "----------------------------------------"
echo "Evaluating t5gemma2-270m on molweni with focus structure"
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_focus.txt
echo "----------------------------------------"
echo "Evaluating t5gemma2-1b on molweni with focus structure"
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_focus.txt
echo "----------------------------------------"
echo "Evaluating t5gemma2-4b on molweni with focus structure"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s focus --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_focus.txt
echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on molweni with natural2 structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s natural2 --lr 2e-5 --seed 27 > eval/t0-3b-3b_molweni_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on molweni with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on molweni with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on molweni with natural2 structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on molweni with natural2 structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5-3b_molweni_natural2.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on molweni with natural2 structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5-large_molweni_natural2.txt
# echo "----------------------------------------"
echo "Evaluating t5gemma2-270m on molweni with natural2 structure"
python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_natural2.txt
echo "----------------------------------------"
echo "Evaluating t5gemma2-1b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_natural2.txt
echo "----------------------------------------"
echo "Evaluating t5gemma2-4b on molweni with natural2 structure"
python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural2 --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_natural2.txt
echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on molweni with augmented structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s augmented --lr 2e-5 --seed 27 > eval/t0-3b-3b_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on molweni with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on molweni with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on molweni with augmented structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on molweni with augmented structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5-3b_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on molweni with augmented structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5-large_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on molweni with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on molweni with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on molweni with augmented structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s augmented --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_augmented.txt
# echo "----------------------------------------"
# echo "Evaluating t0-3b-3b on molweni with natural structure"
# python3 eval_gen.py --fted_model t0-3b-3b --train_corpus molweni --test_corpus molweni -s natural --lr 2e-5 --seed 27 > eval/t0-3b-3b_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-base on molweni with natural structure"
# python3 eval_gen.py --fted_model flan-t5-base --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-base_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-large on molweni with natural structure"
# python3 eval_gen.py --fted_model flan-t5-large --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-large_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating flan-t5-xl on molweni with natural structure"
# python3 eval_gen.py --fted_model flan-t5-xl --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/flan-t5-xl_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5-3b on molweni with natural structure"
# python3 eval_gen.py --fted_model t5-3b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5-3b_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5-large on molweni with natural structure"
# python3 eval_gen.py --fted_model t5-large --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5-large_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-270m on molweni with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-270m --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-270m_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-1b on molweni with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-1b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-1b_molweni_natural.txt
# echo "----------------------------------------"
# echo "Evaluating t5gemma2-4b on molweni with natural structure"
# python3 eval_gen.py --fted_model t5gemma2-4b --train_corpus molweni --test_corpus molweni -s natural --lr 5e-5 --seed 27 > eval/t5gemma2-4b_molweni_natural.txt
# echo "----------------------------------------"