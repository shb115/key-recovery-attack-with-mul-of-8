# Key-Recovery Attack on 5-Round AES with Multiple-of-8 Property

This repository contains the official implementation for the paper: **"Key-Recovery Attack on 5-Round AES with Multiple-of-8 Property"**.

[![Journal Badge](https://img.shields.io/badge/Journal-IEICE_Transactions-blue)](https://www.jstage.jst.go.jp/article/transfun/advpub/0/advpub_2025EAP1069/_article/-char/ja/)

---

## 📝 Abstract

This research proposes a key-recovery attack on 5-round AES (Advanced Encryption Standard) by leveraging the "Multiple-of-8 Property". Our attack can recover the full 128-bit key of 5-round AES with a **time complexity of $2^{34.5}$ 5-round AES encryptions**, a **data complexity of $2^{34.2}$ chosen plaintexts**, and a **memory complexity of $2^{31}$ 128-bit blocks**.

---

## 📂 Repository Structure
├── ciphertools/ # Contains the built .dll and .so shared library files. ├── ciphertoolsL/ # C project for Linux environments. ├── ciphertoolsw/ # C project for Windows environments. ├── lib_source/ # C source code for the AES and attack algorithms. ├── results/ # Pickle files containing experiment results. ├── wrapper/ # Python wrapper scripts that utilize the C library. └── ciphertools.sln # Visual Studio solution file.

-   **`ciphertools.sln`**: The main Visual Studio solution file containing the `ciphertoolsw` (Windows) and `ciphertoolsL` (Linux) projects.
-   **`lib_source`**: Contains the C source code implementing the core AES encryption and key-recovery attack logic.
-   **`ciphertoolsw` / `ciphertoolsL`**: Project files for compiling the C code in Windows and Linux environments, respectively.
-   **`ciphertools`**: The output directory where the compiled shared libraries (`.dll` or `.so`) are placed after a successful build.
-   **`wrapper`**: Contains Python scripts that act as a wrapper to call the functions in the compiled C library and run the attack.
-   **`results`**: Stores `.pickle` files that contain the results from the experiments executed via the Python scripts.

---

## 🛠️ Build Instructions

1.  Open `ciphertools.sln` in Visual Studio.
2.  Select the appropriate project for your environment (`ciphertoolsw` for Windows or `ciphertoolsL` for Linux).
3.  **Build** the project to generate the shared library file (`.dll` for Windows, `.so` for Linux).
4.  Upon a successful build, the corresponding library file will be created in the `ciphertools/` directory.
5.  The generated library can then be imported and used by the Python scripts located in the `wrapper/` directory to execute the attack.

---

## 📄 Citation

If you reference this work, please cite the original paper:

**Authors**: Hanbeom SHIN, Sunyeop KIM, Byoungjin SEOK, Dongjae LEE, Deukjo HONG, Jaechul SUNG, and Seokhie HONG

**Title**: Key-Recovery Attack on 5-Round AES with Multiple-of-8 Property

**Journal**: IEICE TRANSACTIONS on Fundamentals of Electronics, Communications and Computer Sciences (2025)

**DOI**: `https://doi.org/10.1587/transfun.2025EAP1069`

---

## 📜 License

This project is distributed under the [MIT License](LICENSE).
