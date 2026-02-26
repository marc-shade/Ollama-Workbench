# security/encryption.py

"""
Enhanced Encryption Module
Implements AES-256-GCM encryption, key management, and data protection.
"""

import os
import base64
import secrets
import hashlib
import logging
from typing import Union, Tuple, Optional, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import json
from datetime import datetime, timedelta, timezone
from .security_config import get_security_config, get_security_config_manager

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Enhanced encryption manager with AES-256-GCM and RSA support"""
    
    def __init__(self):
        self.config = get_security_config()
        self.config_manager = get_security_config_manager()
        self.backend = default_backend()
        
        # Initialize encryption keys
        self._init_keys()
    
    def _init_keys(self) -> None:
        """Initialize encryption keys"""
        # Get or generate master key
        master_key = self.config_manager.get_secret('encryption_key')
        if not master_key:
            master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
            self.config_manager.set_secret('encryption_key', master_key)
        
        self.master_key = base64.urlsafe_b64decode(master_key.encode('utf-8'))
        
        # Generate or load RSA keys for asymmetric encryption
        self._init_rsa_keys()
    
    def _init_rsa_keys(self) -> None:
        """Initialize RSA key pair"""
        private_key_pem = self.config_manager.get_secret('rsa_private_key')
        public_key_pem = self.config_manager.get_secret('rsa_public_key')
        
        if not private_key_pem or not public_key_pem:
            # Generate new RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=self.backend
            )
            
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_key_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_key_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Store keys
            self.config_manager.set_secret('rsa_private_key', private_pem.decode('utf-8'))
            self.config_manager.set_secret('rsa_public_key', public_pem.decode('utf-8'))
            
            self.rsa_private_key = private_key
            self.rsa_public_key = public_key
        else:
            # Load existing keys
            self.rsa_private_key = serialization.load_pem_private_key(
                private_key_pem.encode('utf-8'),
                password=None,
                backend=self.backend
            )
            
            self.rsa_public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=self.backend
            )
    
    def generate_key(self, password: Optional[str] = None, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password or use master key"""
        if password:
            if salt is None:
                salt = secrets.token_bytes(16)
            
            kdf = PBKDF2HMAC(
                algorithm=SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=self.backend
            )
            
            return kdf.derive(password.encode('utf-8'))
        else:
            return self.master_key
    
    def encrypt_data(self, data: Union[str, bytes], password: Optional[str] = None, 
                    key: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-256-GCM
        
        Returns:
            Tuple of (encrypted_data, nonce, salt_or_empty)
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Generate key
        salt = b''
        if key is None:
            if password:
                salt = secrets.token_bytes(16)
                key = self.generate_key(password, salt)
            else:
                key = self.master_key
        
        # Generate nonce
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce),
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine ciphertext and authentication tag
        encrypted_data = ciphertext + encryptor.tag
        
        return encrypted_data, nonce, salt
    
    def decrypt_data(self, encrypted_data: bytes, nonce: bytes, password: Optional[str] = None,
                    key: Optional[bytes] = None, salt: Optional[bytes] = None) -> bytes:
        """
        Decrypt data using AES-256-GCM
        """
        # Generate key
        if key is None:
            if password and salt:
                key = self.generate_key(password, salt)
            else:
                key = self.master_key
        
        # Split ciphertext and authentication tag
        ciphertext = encrypted_data[:-16]  # All but last 16 bytes
        tag = encrypted_data[-16:]  # Last 16 bytes
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def encrypt_string(self, text: str, password: Optional[str] = None) -> str:
        """Encrypt string and return base64-encoded result"""
        encrypted_data, nonce, salt = self.encrypt_data(text, password)
        
        # Create container with metadata
        container = {
            'data': base64.b64encode(encrypted_data).decode('utf-8'),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'salt': base64.b64encode(salt).decode('utf-8') if salt else '',
            'algorithm': 'AES-256-GCM',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return base64.b64encode(json.dumps(container).encode('utf-8')).decode('utf-8')
    
    def decrypt_string(self, encrypted_text: str, password: Optional[str] = None) -> str:
        """Decrypt base64-encoded encrypted string"""
        try:
            # Decode container
            container_json = base64.b64decode(encrypted_text.encode('utf-8')).decode('utf-8')
            container = json.loads(container_json)
            
            # Extract components
            encrypted_data = base64.b64decode(container['data'].encode('utf-8'))
            nonce = base64.b64decode(container['nonce'].encode('utf-8'))
            salt = base64.b64decode(container['salt'].encode('utf-8')) if container['salt'] else None
            
            # Decrypt
            plaintext = self.decrypt_data(encrypted_data, nonce, password, salt=salt)
            return plaintext.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def encrypt_file(self, filepath: str, output_filepath: Optional[str] = None,
                    password: Optional[str] = None) -> str:
        """Encrypt file and save to disk"""
        if output_filepath is None:
            output_filepath = filepath + '.encrypted'
        
        try:
            # Read file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Encrypt
            encrypted_data, nonce, salt = self.encrypt_data(data, password)
            
            # Create metadata
            metadata = {
                'algorithm': 'AES-256-GCM',
                'original_filename': os.path.basename(filepath),
                'encrypted_at': datetime.now(timezone.utc).isoformat(),
                'nonce': base64.b64encode(nonce).decode('utf-8'),
                'salt': base64.b64encode(salt).decode('utf-8') if salt else '',
                'data_size': len(encrypted_data)
            }
            
            # Write encrypted file
            with open(output_filepath, 'wb') as f:
                # Write metadata length (4 bytes)
                metadata_json = json.dumps(metadata).encode('utf-8')
                f.write(len(metadata_json).to_bytes(4, 'big'))
                
                # Write metadata
                f.write(metadata_json)
                
                # Write encrypted data
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {filepath} -> {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise
    
    def decrypt_file(self, encrypted_filepath: str, output_filepath: Optional[str] = None,
                    password: Optional[str] = None) -> str:
        """Decrypt file and save to disk"""
        try:
            # Read encrypted file
            with open(encrypted_filepath, 'rb') as f:
                # Read metadata length
                metadata_length = int.from_bytes(f.read(4), 'big')
                
                # Read metadata
                metadata_json = f.read(metadata_length).decode('utf-8')
                metadata = json.loads(metadata_json)
                
                # Read encrypted data
                encrypted_data = f.read()
            
            # Extract components
            nonce = base64.b64decode(metadata['nonce'].encode('utf-8'))
            salt = base64.b64decode(metadata['salt'].encode('utf-8')) if metadata['salt'] else None
            
            # Decrypt
            plaintext = self.decrypt_data(encrypted_data, nonce, password, salt=salt)
            
            # Determine output filename
            if output_filepath is None:
                output_filepath = metadata.get('original_filename', 'decrypted_file')
            
            # Write decrypted file
            with open(output_filepath, 'wb') as f:
                f.write(plaintext)
            
            logger.info(f"File decrypted: {encrypted_filepath} -> {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise
    
    def encrypt_rsa(self, data: Union[str, bytes], use_public_key: bool = True) -> bytes:
        """Encrypt data using RSA (for small data like keys)"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = self.rsa_public_key if use_public_key else self.rsa_private_key
        
        encrypted = key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=SHA256()),
                algorithm=SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def decrypt_rsa(self, encrypted_data: bytes, use_private_key: bool = True) -> bytes:
        """Decrypt data using RSA"""
        key = self.rsa_private_key if use_private_key else self.rsa_public_key
        
        decrypted = key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=SHA256()),
                algorithm=SHA256(),
                label=None
            )
        )
        
        return decrypted
    
    def secure_hash(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """Generate secure hash of data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if algorithm == 'sha256':
            hash_obj = hashlib.sha256(data)
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512(data)
        elif algorithm == 'blake2b':
            hash_obj = hashlib.blake2b(data)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
    
    def rotate_keys(self) -> None:
        """Rotate encryption keys"""
        # Generate new master key
        new_master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
        old_master_key = self.config_manager.get_secret('encryption_key')
        
        # Update master key
        self.config_manager.set_secret('encryption_key', new_master_key)
        self.master_key = base64.urlsafe_b64decode(new_master_key.encode('utf-8'))
        
        # Generate new RSA keys
        self._init_rsa_keys()
        
        logger.info("Encryption keys rotated successfully")
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        return self.config_manager.get_secret('rsa_public_key')
    
    def verify_integrity(self, data: bytes, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Verify data integrity using hash"""
        actual_hash = self.secure_hash(data, algorithm)
        return actual_hash == expected_hash

# Global encryption manager
_encryption_manager = None

def get_encryption_manager() -> EncryptionManager:
    """Get the global encryption manager"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager

def encrypt_data(data: Union[str, bytes], password: Optional[str] = None) -> str:
    """Encrypt data - convenience function"""
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    return get_encryption_manager().encrypt_string(data, password)

def decrypt_data(encrypted_data: str, password: Optional[str] = None) -> str:
    """Decrypt data - convenience function"""
    return get_encryption_manager().decrypt_string(encrypted_data, password)

def generate_encryption_key() -> str:
    """Generate new encryption key"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')

def secure_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """Generate secure hash - convenience function"""
    return get_encryption_manager().secure_hash(data, algorithm)