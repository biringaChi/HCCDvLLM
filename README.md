Installation
```
git clone https://github.com/biringaChi/M2
```

Dependencies
```
pip install -r requirements.txt
```

Core directory
```
cd core
```

Run M1 model
```
python detect.py --model=M1 --epochs=10 --lr=1e-3
```

Run M2 model
```
python detect.py --model=M2 --epochs=4 --lr=1e-4
```