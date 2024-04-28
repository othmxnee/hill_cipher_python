
import numpy as np
from egcd import egcd 

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 !#$%&'()*+,-./:;<=>?@[\]^_`{|}~√⇝∞▶▲░◀●☀☢☠☟☞☝☜☛☚☑☐☉☆★☄☃☂☁☣☪☮☯☸☹☺☻☼☽☾♔♕♖♗♘♚♛♜❝❞❣❤❥❦⌨︎✏︎✒︎✉︎✂︎⛏︎⚒︎⚔︎⚙︎⚖︎⭣⭤⭥⮂⮃⮐⮑⬎⬏⬐⬑⬱⬳⬸⬿⭅⭆↕↖↗↘↙↚↛↜↝↞↟↠↡↢↣↤↥↦↧↨↩↪↫↬↭↮↯↰↱↲↳↴↶↷↸↹↺↻⟲⟳↼↽↾↿⇀⇁⇂⇃⇄⇅⇆⇇⇈⇉⇊⇋⇌⇍⇎⇏⇕⇖⇗⇘⇙⇚⇛⇜⇝⇞⇟⇠⇡⇢"
print("-----------------------------------")
print("Length of the alphabet is : ")
print(len(alphabet))
print("-----------------------------------")
#convert letter to number and the opposite
letter_to_index = dict(zip(alphabet, range(len(alphabet))))
index_to_letter = dict(zip(range(len(alphabet)), alphabet))
#invert the matrix
def matrix_mod_inv(matrix, modulus):
    if np.linalg.det(matrix)==0:
        print("determinant of matrix is null")
    det = int(np.round(np.linalg.det(matrix)))  
    det_inv = egcd(det, modulus)[1] % modulus  
    matrix_modulus_inv = (
        det_inv * np.round(det * np.linalg.inv(matrix)).astype(int) % modulus
    ) 
    return matrix_modulus_inv

#encryption
def encrypt(message, K):
    cipher_text = ""
    message_in_numbers = []

    for letter in message:
        message_in_numbers.append(letter_to_index[letter])
#splitting
    split_P = [
        message_in_numbers[i : i + int(2)]
        for i in range(0, len(message_in_numbers), int(2))
    ]
    for P in split_P:
        P = np.transpose(np.asarray(P))[:, np.newaxis]

        if P.shape[0] != 2:
            P = np.append(P, letter_to_index["Z"])[:, np.newaxis]
        
#multiplication
        numbers = np.dot(K, P) % 256
        n = numbers.shape[0]  

        for idx in range(n):
            number = int(numbers[idx, 0])
            cipher_text += index_to_letter[number]

    return cipher_text

#decryption
def decrypt(cipher, Key_inv):
    plain_text = ""
    cipher_in_numbers = []

    for letter in cipher:
        cipher_in_numbers.append(letter_to_index[letter])
#splitting
    split_C = [
        cipher_in_numbers[i : i + 2]
        for i in range(0, len(cipher_in_numbers), 2)
    ]
    
    for C in split_C:
        C = np.transpose(np.asarray(C))[:, np.newaxis]
        numbers = np.dot(Key_inv, C) % 256
        n = numbers.shape[0]

        for idx in range(n):
            number = int(numbers[idx, 0])
            plain_text += index_to_letter[number]
   
    

    return plain_text
#Known plain text attack
def attack():
    c = "↴⟲↥↼"  
    m = "take"  
# k = c * m_inv % 256
    if len(c) != 4 or len(m) != 4:
        print("Cipher-text and plaintext must each be 4 characters long for a 2x2 matrix.")
        return

    # convert characters to numbers
    c_in_numbers = np.array([letter_to_index[char] for char in c]).reshape(2, 2).T
    m_in_numbers = np.array([letter_to_index[char] for char in m]).reshape(2, 2).T
    print("About the attack : ")
    print("-----------------------------------")
    print("Plain-text Matrix (M):\n", m_in_numbers)
    print("-----------------------------------")
    print("Cipher-text Matrix (C):\n", c_in_numbers)
    print("-----------------------------------")

    try:
        M_inv = np.linalg.inv(m_in_numbers)
        print("Inverse of M:\n", M_inv)
        print("-----------------------------------")

        # Compute the key matrix K 
        K = np.dot(c_in_numbers, M_inv) % len(alphabet)
        
        
        print("Recovered Key Matrix K:\n", K)
        print("-----------------------------------")
    except Exception as e:
        print("Error in computing the inverse or the key:", e)




    


def main():
    
    #message = "take"
    message = 'take your time broo'

    K = np.matrix([[3, 3], [2, 5]])
    
    Key_inv = matrix_mod_inv(K, len(alphabet))

    encrypted_message = encrypt(message, K)
    plain_text_message = decrypt(encrypted_message, Key_inv)

    print("The original message is : " + message)
    print("-----------------------------------")
    print("The cipher_text is : " + encrypted_message)
    print("-----------------------------------")
    print("The plain_text message after decryption is : " + plain_text_message)
    print("-----------------------------------")
    attack()

main()