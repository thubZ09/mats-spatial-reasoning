import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random
import time

from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MATSExample:
    """single MATS example with image and metadata"""
    example_id: str
    image: Image.Image
    statement: str
    relation: str
    is_true: bool
    objects: List[str]
    split: str = "test"
    
    def to_dict(self) -> Dict:
        """convert to dictionary for JSON serialization"""
        return {
            'example_id': self.example_id,
            'statement': self.statement,
            'relation': self.relation,
            'is_true': self.is_true,
            'objects': self.objects,
            'split': self.split
        }


class MATSDatasetManager:
    """manages dataset creation from VSR HuggingFace dataset"""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        download_images: bool = True
    ):
        """
        Args:
            cache_dir: directory for caching HF dataset
            download_images: whether to download COCO images (True recommended)
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "mats")
        self.download_images = download_images
        
        self.target_relations = ['left', 'right', 'above', 'below', 'front', 'behind'] #spatial relations
          
        #VSR to MATS relation mapping
        self.relation_mapping = {
            'left of': 'left',
            'at the left side of': 'left',
            'right of': 'right',
            'at the right side of': 'right',
            'above': 'above',
            'on top of': 'above',
            'over': 'above',
            'below': 'below',
            'under': 'below',
            'beneath': 'below',
            'in front of': 'front',
            'behind': 'behind',
        }
        
        #stats tracking
        self.stats = {
            'processed': 0,
            'wrong_label': 0,
            'wrong_relation': 0,
            'download_failed': 0,
            'bucket_full': 0,
            'success': 0
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_vsr_from_huggingface(self, use_all_splits: bool = True) -> List[Dict]:
        """
        load VSR dataset from HuggingFace
        
        Args:
            use_all_splits: If True, load train+val+test     
        Returns:
            list of VSR examples
        """
        logger.info(f"Loading VSR dataset from HuggingFace..")
        
        try:
            all_data = []
            
            if use_all_splits:
                for split_name in ["train", "validation", "test"]:
                    logger.info(f"  Loading {split_name} split...")
                    dataset = load_dataset(
                        "cambridgeltl/vsr_random",
                        split=split_name,
                        cache_dir=self.cache_dir
                    )
                    all_data.extend(list(dataset))
                    logger.info(f" {len(dataset)} examples from {split_name}")
            else:
                dataset = load_dataset(
                    "cambridgeltl/vsr_random",
                    split="train",
                    cache_dir=self.cache_dir
                )
                all_data = list(dataset)
            
            logger.info(f"Loaded {len(all_data)} total VSR examples from HuggingFace")
            return all_data
            
        except Exception as e:
            logger.error(f"Failed to load VSR from HuggingFace: {e}")
            raise
    
    def download_coco_image(
        self, 
        image_url: str, 
        max_retries: int = 3, 
        timeout: int = 30,
        retry_delay: float = 1.0
    ) -> Optional[Image.Image]:
        """
        download COCO image from URL with retry logic
        
        Args:
            image_url: COCO image URL
            max_retries: number of retry attempts
            timeout: request timeout in seconds
            retry_delay: seconds to wait between retries    
        Returns:
            PIL Image or None if failed
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(image_url, timeout=timeout)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                return image
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.debug(f"Timeout on attempt {attempt+1}/{max_retries} for {image_url}, retrying...")
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"Error on attempt {attempt+1}/{max_retries} for {image_url}: {e}, retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.debug(f"Failed to download {image_url} after {max_retries} attempts: {e}")
        
        return None
    
    def normalize_relation(self, vsr_relation: str) -> Optional[str]:
        """
        map VSR relation to MATS target relation
        
        Args:
            vsr_relation: original VSR relation string  
        Returns:
            normalized relation or None if not in target set
        """
        normalized = self.relation_mapping.get(vsr_relation, vsr_relation)
        return normalized if normalized in self.target_relations else None
    
    def create_vsr_split(
        self,
        n_per_relation: int = 50,
        only_true_labels: bool = True,
        max_attempts: Optional[int] = None
    ) -> List[MATSExample]:
        """
        create MATS VSR split with balanced relations
        
        Args:
            n_per_relation: number of examples per spatial relation (default: 50)
            only_true_labels: only use TRUE examples (recommended for SCS)
            max_attempts: maximum VSR examples to scan   
        Returns:
            list of MATS examples
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Creating MATS VSR split: {n_per_relation} examples per relation")
        logger.info(f"Target total: {n_per_relation * len(self.target_relations)} examples")
        logger.info(f"{'='*70}\n")
        
        vsr_data = self.load_vsr_from_huggingface(use_all_splits=True)  #load VSR data from ALL splits
        
        self.stats = {k: 0 for k in self.stats}  #reset stats
        
        relation_buckets = {rel: [] for rel in self.target_relations}  #collect examples by relation
        random.shuffle(vsr_data)  #shuffle for random sampling
        
        #progress tracking
        total_needed = n_per_relation * len(self.target_relations)
        last_progress = 0
        
        for idx, item in enumerate(vsr_data):
            total_collected = sum(len(bucket) for bucket in relation_buckets.values())
            if total_collected >= total_needed:
                logger.info(f"\nCollected all {total_needed} examples!")
                break
            
            #progress update every 10%
            if max_attempts and idx % max(1, max_attempts // 10) == 0:
                progress = (idx / len(vsr_data)) * 100
                if progress - last_progress >= 10:
                    logger.info(f"Progress: {idx}/{len(vsr_data)} ({progress:.1f}%) - Collected: {total_collected}/{total_needed}")
                    logger.info(f"  Relations: {[(rel, len(bucket)) for rel, bucket in relation_buckets.items()]}")
                    last_progress = progress
            
            self.stats['processed'] += 1
            
            #filter: only TRUE labels if specified
            if only_true_labels and item['label'] != 1:
                self.stats['wrong_label'] += 1
                continue
            
            #normalize relation
            normalized_rel = self.normalize_relation(item['relation'])
            if normalized_rel is None:
                self.stats['wrong_relation'] += 1
                continue
            
            #check if we need more of this relation
            if len(relation_buckets[normalized_rel]) >= n_per_relation:
                self.stats['bucket_full'] += 1
                continue
            
            #download image with retries
            if self.download_images:
                image = self.download_coco_image(
                    item['image_link'],
                    max_retries=3,
                    timeout=30,
                    retry_delay=1.0
                )
                if image is None:
                    self.stats['download_failed'] += 1
                    continue
            else:
                image = Image.new('RGB', (512, 512), color=(240, 240, 240))
             
            objects = self._extract_objects_from_caption(item['caption'])  #extract objects from caption
            
            #create MATS example
            example = MATSExample(
                example_id=f"vsr_{item['image'].replace('.jpg', '')}",
                image=image,
                statement=item['caption'],
                relation=normalized_rel,
                is_true=(item['label'] == 1),
                objects=objects,
                split="test"
            )
            
            relation_buckets[normalized_rel].append(example)
            self.stats['success'] += 1
            
            #check if done
            if all(len(bucket) >= n_per_relation for bucket in relation_buckets.values()):
                logger.info(f"\nCollected all required examples at index {idx}!")
                break
        
        #flatten all buckets
        all_examples = []
        for rel, bucket in relation_buckets.items():
            collected = len(bucket[:n_per_relation])
            all_examples.extend(bucket[:n_per_relation])
            
            if collected < n_per_relation:
                logger.warning(f"{rel}: Only {collected}/{n_per_relation} examples (SHORT!)")
            else:
                logger.info(f"{rel}: {collected}/{n_per_relation} examples")
        
        #print statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"VSR Creation Statistics:")
        logger.info(f"{'='*70}")
        logger.info(f"Processed: {self.stats['processed']} examples")
        logger.info(f" Success: {self.stats['success']}")
        logger.info(f" Skipped:")
        logger.info(f"   - Wrong label: {self.stats['wrong_label']}")
        logger.info(f"   - Wrong relation: {self.stats['wrong_relation']}")
        logger.info(f"   - Download failed: {self.stats['download_failed']}")
        logger.info(f"   - Bucket full: {self.stats['bucket_full']}")
        logger.info(f"\n Final: {len(all_examples)}/{total_needed} examples collected")
        logger.info(f"{'='*70}\n")
        
        if len(all_examples) < total_needed * 0.9:
            logger.warning(f" WARNING: Only collected {len(all_examples)}/{total_needed} examples (< 90%)")
            logger.warning(f"   Consider:")
            logger.warning(f"   1. Check internet connection")
            logger.warning(f"   2. COCO images may be unavailable")
            logger.warning(f"   3. Try again later")
        
        return all_examples
    
    def create_absurd_pairs(
        self,
        vsr_examples: List[MATSExample],
        n_pairs: int = 300
    ) -> List[Dict]:
        """        
        Args:
            vsr_examples: list of VSR examples
            n_pairs: number of absurd pairs to create    
        Returns:
            list of absurd pair dicts (with unified schema)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Creating {n_pairs} absurd pairs...")
        logger.info(f"{'='*70}\n")
        
        #inversion mapping
        inversions = {
            'left': 'right',
            'right': 'left',
            'above': 'below',
            'below': 'above',
            'front': 'behind',
            'behind': 'front'
        }
        
        #filter to invertible examples
        invertible = [ex for ex in vsr_examples if ex.relation in inversions]
        
        if len(invertible) < n_pairs:
            logger.warning(f"Only {len(invertible)} invertible examples available (need {n_pairs})")
            logger.warning(f"   Will create {len(invertible)} absurd pairs instead")
            n_pairs = len(invertible)
        
        absurd_pairs = []
        sampled = random.sample(invertible, n_pairs)
        
        for i, example in enumerate(sampled):
            #invert relation in statement
            inverted_rel = inversions[example.relation]
            absurd_statement = example.statement.replace(example.relation, inverted_rel)
            
            #use unified schema matching VSR
            absurd_pairs.append({
                'example_id': f"absurd_{i:04d}",
                'image': example.image,
                'statement': absurd_statement,
                'relation': inverted_rel,  
                'is_true': False, 
                'objects': example.objects,
                'split': 'absurd',
                #additional metadata 
                'metadata': json.dumps({
                    'category': 'spatial',
                    'is_absurd': True,
                    'original_statement': example.statement,
                    'original_relation': example.relation
                })
            })
        
        logger.info(f"Created {len(absurd_pairs)} absurd pairs\n")
        return absurd_pairs
    
    def create_patching_pairs(
        self,
        vsr_examples: List[MATSExample],
        n_pairs: int = 420,
        allow_resampling: bool = True
    ) -> List[Dict]:
        """
        donor-target pairs for activation patching
        
        Args:
            vsr_examples: list of VSR examples
            n_pairs: number of patching pairs to create
            allow_resampling: allow sampling same example multiple times    
        Returns:
            list of patching pair dicts (with unified schema)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Creating {n_pairs} patching pairs...")
        logger.info(f"{'='*70}\n")
        
        inversions = {
            'left': 'right',
            'right': 'left',
            'above': 'below',
            'below': 'above',
            'front': 'behind',
            'behind': 'front'
        }
        
        #filter to invertible relations
        invertible_examples = [
            ex for ex in vsr_examples 
            if ex.relation in inversions
        ]
        
        logger.info(f"Available invertible examples: {len(invertible_examples)}")
        
        if len(invertible_examples) < n_pairs:
            if allow_resampling:
                logger.info(f"Using resampling to create {n_pairs} pairs from {len(invertible_examples)} examples")
                sampled = random.choices(invertible_examples, k=n_pairs) #sample with replacement
            else: 
                logger.warning(f"Only {len(invertible_examples)} invertible examples available (need {n_pairs})")
                logger.warning(f"Will create {len(invertible_examples)} patching pairs instead")
                n_pairs = len(invertible_examples)
                sampled = invertible_examples
        else:
            sampled = random.sample(invertible_examples, n_pairs)  #sample without replacement
        
        patching_pairs = []
        
        for i, example in enumerate(sampled):
            inverted_rel = inversions[example.relation]
            absurd_statement = example.statement.replace(example.relation, inverted_rel)
            
            #use unified schema with metadata for patching-specific fields
            patching_pairs.append({
                'example_id': f"patch_{i:04d}",
                'image': example.image, 
                'statement': example.statement,  
                'relation': example.relation,  
                'is_true': True,  
                'objects': example.objects,
                'split': 'patching',
                #patching-specific data stored in metadata
                'metadata': json.dumps({
                    'absurd_text': absurd_statement,
                    'inverted_relation': inverted_rel,
                    'has_donor_target_pair': True
                })
            })
        
        logger.info(f"Created {len(patching_pairs)} patching pairs\n")
        return patching_pairs
    
    def save_for_huggingface(
        self,
        vsr_examples: List[MATSExample],
        absurd_pairs: List[Dict],
        patching_pairs: List[Dict],
        output_dir: str = "mats_dataset_export"
    ):
        """
        format ready for HuggingFace upload
        
        Args:
            vsr_examples: VSR examples
            absurd_pairs: Absurd pairs
            patching_pairs: Patching pairs
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        #save images
        images_dir = output_path / "images"
        (images_dir / "vsr").mkdir(exist_ok=True, parents=True)
        (images_dir / "absurd").mkdir(exist_ok=True, parents=True)
        (images_dir / "patching").mkdir(exist_ok=True, parents=True)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Saving dataset to {output_path}...")
        logger.info(f"{'='*70}\n")
        
        #save VSR with unified schema
        logger.info("Saving VSR examples...")
        vsr_json = []
        for ex in vsr_examples:
            img_filename = f"{ex.example_id}.jpg"
            ex.image.save(images_dir / "vsr" / img_filename)
            
            vsr_json.append({
                'example_id': ex.example_id,
                'image': img_filename,
                'statement': ex.statement,
                'relation': ex.relation,
                'is_true': ex.is_true,
                'objects': ex.objects,
                'split': ex.split,
                'metadata': json.dumps({})  #empty metadata for VSR
            })
        
        with open(output_path / "vsr.json", 'w') as f:
            json.dump(vsr_json, f, indent=2)
        logger.info(f"Saved {len(vsr_json)} VSR examples")
        
        #save absurd 
        logger.info("Saving Absurd pairs...")
        absurd_json = []
        for pair in absurd_pairs:
            img_filename = f"{pair['example_id']}.jpg"
            pair['image'].save(images_dir / "absurd" / img_filename)
            
            absurd_json.append({
                'example_id': pair['example_id'],
                'image': img_filename,
                'statement': pair['statement'],
                'relation': pair['relation'],
                'is_true': pair['is_true'],
                'objects': pair['objects'],
                'split': pair['split'],
                'metadata': pair['metadata']
            })
        
        with open(output_path / "absurd.json", 'w') as f:
            json.dump(absurd_json, f, indent=2)
        logger.info(f" Saved {len(absurd_json)} Absurd pairs")
        
        #save patching
        logger.info("Saving Patching pairs...")
        patching_json = []
        for pair in patching_pairs:
            img_filename = f"{pair['example_id']}.jpg"
            pair['image'].save(images_dir / "patching" / img_filename)
            
            patching_json.append({
                'example_id': pair['example_id'],
                'image': img_filename,
                'statement': pair['statement'],
                'relation': pair['relation'],
                'is_true': pair['is_true'],
                'objects': pair['objects'],
                'split': pair['split'],
                'metadata': pair['metadata']
            })
        
        with open(output_path / "patching.json", 'w') as f:
            json.dump(patching_json, f, indent=2)
        logger.info(f" Saved {len(patching_json)} Patching pairs")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Dataset saved successfully to {output_path}")
        logger.info(f"{'='*70}")
    
    def _extract_objects_from_caption(self, caption: str) -> List[str]:
        """extract objects from VSR caption"""
        #remove common words
        stopwords = {'the', 'is', 'of', 'a', 'an', 'to', 'in', 'on', 'at', 'from'}
        words = caption.lower().replace('.', '').split()
        
        objects = [w for w in words if w not in stopwords and len(w) > 2] #keep nouns

        return objects[:3] if len(objects) >= 2 else objects #return first 2-3 objects


#convienence functions
def load_mats_vsr(n_per_relation: int = 50, download_images: bool = True) -> List[MATSExample]:
    """
    quick function to load MATS VSR dataset
    
    Args:
        n_per_relation: examples per spatial relation (default: 50 → 300 total)
        download_images: download COCO images from web  
    Returns:
        list of MATS examples
    
    Example:
        >>> vsr_data = load_mats_vsr(n_per_relation=50)
        >>> print(f"Loaded {len(vsr_data)} VSR examples")
    """
    manager = MATSDatasetManager(download_images=download_images)
    return manager.create_vsr_split(n_per_relation=n_per_relation)

def create_full_mats_dataset(
    output_dir: str = "mats_dataset_export",
    n_vsr_per_relation: int = 50,
    n_absurd: int = 300,
    n_patching: int = 420
) -> Tuple[List[MATSExample], List[Dict], List[Dict]]:
    """   
    Args:
        output_dir: where to save the dataset
        n_vsr_per_relation: VSR examples per relation (50 → 300 total)
        n_absurd: number of absurd pairs
        n_patching: number of patching pairs   
    Returns:
        (vsr_examples, absurd_pairs, patching_pairs)
    
    Example:
        >>> vsr, absurd, patching = create_full_mats_dataset()
        >>> # Now upload to HuggingFace using scripts/5_upload_to_hf.py
    """
    logger.info("\n" + "="*70)
    logger.info("CREATING COMPLETE MATS DATASET FROM VSR")
    logger.info("="*70 + "\n")
    
    manager = MATSDatasetManager(download_images=True)

    vsr_examples = manager.create_vsr_split(n_per_relation=n_vsr_per_relation) #create VSR split
    absurd_pairs = manager.create_absurd_pairs(vsr_examples, n_pairs=n_absurd) #create absurd pairs
    patching_pairs = manager.create_patching_pairs(vsr_examples, n_pairs=n_patching)  #create patching pairs
    
    manager.save_for_huggingface(vsr_examples, absurd_pairs, patching_pairs, output_dir) #save for HuggingFace
    
    return vsr_examples, absurd_pairs, patching_pairs