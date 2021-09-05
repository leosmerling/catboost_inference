# Catboost prediction from rust

> This is a test calling catboost predictions from rust to compare performance and check memory leaks


For unoptimized:
```
cargo run
```

For release mode
```
./run.sh
```


Num iterations and batch size can be changed in `main.rs` file.


To check for memory leaks, run valgrind with different iterations and compare results:

```
valgrind --tool=memcheck --leak-check=full \       
--log-file=catboost-rust2.valgrind.log --track-origins=yes --show-leak-kinds=all \
./run.sh
```
