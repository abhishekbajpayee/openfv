OpenFV
======

<a href="http://abhishekbajpayee.github.io/openfv/docs/build/html/" target="_blank">OpenFV Documentation</a>

Please let us know or submit a pull request with suggested fixes if you notice any issues with the documentation. 

Roadmap
=======

### Phase 1
- Complete Python bindings for all C++ functionality
- Add tomo reconstruction code (MATLAB)

### Phase 2
- Add tests that ensure functionality is not breaking due to updates
- Remove MATLAB code and call underlying algorithms (in Fortran) from Python
- Eliminate use of Intel MKL (in Fortran code) and replace with FFTW
- Establish a framework for effective benchmarking and comparison of different algorithms using the exact same data / inputs

### Phase 3
- Streamline execution pipeline so that reconstruction via multiple techniques can seamlessly be bound to multiple PIV algorithms for comparison purposes
- Develope API for researchers to hook in custom reconstruction and PIV algorithms for benchmarking