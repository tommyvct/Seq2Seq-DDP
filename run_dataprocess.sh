# Natural (Dependencies for Focus/Natural2)
# python3 dataprocess.py --dataset stac --split train --structure_type natural
# python3 dataprocess.py --dataset stac --split dev --structure_type natural
# python3 dataprocess.py --dataset stac --split test --structure_type natural
python3 dataprocess.py --dataset molweni --split train --structure_type natural
python3 dataprocess.py --dataset molweni --split dev --structure_type natural
python3 dataprocess.py --dataset molweni --split test --structure_type natural
python3 dataprocess.py --dataset discord_unveiled --split train --structure_type natural
python3 dataprocess.py --dataset discord_unveiled --split dev --structure_type natural
python3 dataprocess.py --dataset discord_unveiled --split test --structure_type natural
python3 dataprocess.py --dataset discord_unveiled-hintswap --split train --structure_type natural
python3 dataprocess.py --dataset discord_unveiled-hintswap --split dev --structure_type natural
python3 dataprocess.py --dataset discord_unveiled-hintswap --split test --structure_type natural

# Augmented
# python3 dataprocess.py --dataset stac --split train --structure_type augmented
# python3 dataprocess.py --dataset stac --split dev --structure_type augmented
# python3 dataprocess.py --dataset stac --split test --structure_type augmented
python3 dataprocess.py --dataset molweni --split train --structure_type augmented
python3 dataprocess.py --dataset molweni --split dev --structure_type augmented
python3 dataprocess.py --dataset molweni --split test --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled --split train --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled --split dev --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled --split test --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled-hintswap --split train --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled-hintswap --split dev --structure_type augmented
python3 dataprocess.py --dataset discord_unveiled-hintswap --split test --structure_type augmented


# Transition-based (Focus/Natural2)
# python3 dataprocess.py --dataset stac --split train --structure_type focus
# python3 dataprocess.py --dataset stac --split train --structure_type natural2
# python3 dataprocess.py --dataset stac --split dev --structure_type focus
# python3 dataprocess.py --dataset stac --split dev --structure_type natural2
# python3 dataprocess.py --dataset stac --split test --structure_type focus
# python3 dataprocess.py --dataset stac --split test --structure_type natural2
python3 dataprocess.py --dataset molweni --split train --structure_type focus
python3 dataprocess.py --dataset molweni --split train --structure_type natural2
python3 dataprocess.py --dataset molweni --split dev --structure_type focus
python3 dataprocess.py --dataset molweni --split dev --structure_type natural2
python3 dataprocess.py --dataset molweni --split test --structure_type focus
python3 dataprocess.py --dataset molweni --split test --structure_type natural2

python3 dataprocess.py --dataset discord_unveiled --split train --structure_type focus
python3 dataprocess.py --dataset discord_unveiled --split train --structure_type natural2
python3 dataprocess.py --dataset discord_unveiled --split dev --structure_type focus
python3 dataprocess.py --dataset discord_unveiled --split dev --structure_type natural2
python3 dataprocess.py --dataset discord_unveiled --split test --structure_type focus
python3 dataprocess.py --dataset discord_unveiled --split test --structure_type natural2

python3 dataprocess.py --dataset discord_unveiled-hintswap --split train --structure_type focus
python3 dataprocess.py --dataset discord_unveiled-hintswap --split train --structure_type natural2
python3 dataprocess.py --dataset discord_unveiled-hintswap --split dev --structure_type focus
python3 dataprocess.py --dataset discord_unveiled-hintswap --split dev --structure_type natural2
python3 dataprocess.py --dataset discord_unveiled-hintswap --split test --structure_type focus
python3 dataprocess.py --dataset discord_unveiled-hintswap --split test --structure_type natural2
