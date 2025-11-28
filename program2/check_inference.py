"""
Fast R-CNN Inference Checker
Loads saved data subset and runs inference for visualization and checking.
"""
import argparse
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from functools import partial
import xml.etree.ElementTree as ET

from dataset import idx_to_class, class_to_idx
from model import FastRCNN
from utils import coord_trans, data_visualizer


class SimpleVOCDataset(Dataset):
    """
    Simple dataset for loading saved VOC images and annotations.
    
    Expected directory structure:
    data_dir/
        images/
            000042.jpg
            000032.jpg
            ...
        annotations/
            000042.xml  (Pascal VOC XML format)
            000032.xml
            ...
        proposals/
            000042.json
            000032.json
            ...
    """
    
    def __init__(self, data_dir):
        """
        Args:
            data_dir: Root directory containing images/, annotations/, and proposals/
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.annotations_dir = os.path.join(data_dir, 'annotations')
        self.proposals_dir = os.path.join(data_dir, 'proposals')
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        print(f"Loaded {len(self.image_files)} images from {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_xml_annotation(self, xml_path):
        """
        Parse Pascal VOC XML annotation file.
        
        Returns:
            ann: Dictionary containing annotation information in VOC format
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get size information
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        
        # Get filename
        filename = root.find('filename').text
        
        # Get all objects
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            obj_dict = {
                'name': name,
                'bndbox': {
                    'xmin': bndbox.find('xmin').text,
                    'ymin': bndbox.find('ymin').text,
                    'xmax': bndbox.find('xmax').text,
                    'ymax': bndbox.find('ymax').text
                }
            }
            objects.append(obj_dict)
        
        # Convert to the format expected by collate_fn
        ann = {
            'annotation': {
                'filename': filename,
                'size': {
                    'width': width,
                    'height': height
                },
                'object': objects
            }
        }
        
        return ann
    
    def __getitem__(self, idx):
        """
        Returns:
            img: PIL Image
            ann: Dictionary containing annotation information in VOC format
        """
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Load annotation (XML format)
        ann_file = img_file.replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml')
        ann_path = os.path.join(self.annotations_dir, ann_file)
        
        # Parse XML annotation
        ann = self.parse_xml_annotation(ann_path)
        
        return img, ann


def collate_fn(batch_lst, reshape_size=224, proposal_path=None):
    """
    Collate function for SimpleVOCDataset.
    """
    preprocess = transforms.Compose([
        transforms.Resize((reshape_size, reshape_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    batch_size = len(batch_lst)
    
    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)
    
    box_list = []
    box_batch_idx = []
    w_list = []
    h_list = []
    img_id_list = []
    proposal_list = []
    proposal_batch_idx = []
    
    for i in range(batch_size):
        img, ann = batch_lst[i]
        w_list.append(img.size[0])  # image width
        h_list.append(img.size[1])  # image height
        img_id_list.append(ann['annotation']['filename'])
        img_batch[i] = preprocess(img)
        all_bbox = ann['annotation']['object']
        if type(all_bbox) == dict:  # inconsistency in the annotation file
            all_bbox = [all_bbox]
        for bbox_idx, one_bbox in enumerate(all_bbox):
            bbox = one_bbox['bndbox']
            obj_cls = one_bbox['name']
            box_list.append(torch.Tensor([float(bbox['xmin']), float(bbox['ymin']),
                                         float(bbox['xmax']), float(bbox['ymax']), class_to_idx[obj_cls]]))
            box_batch_idx.append(i)
        if proposal_path is not None:
            proposal_fn = ann['annotation']['filename'].replace('.jpg', '.json')
            with open(os.path.join(proposal_path, proposal_fn), 'r') as f:
                proposal = json.load(f)
            for p in proposal:
                proposal_list.append([p['x_min'], p['y_min'], p['x_max'], p['y_max']])
                proposal_batch_idx.append(i)
    
    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)
    box_batch = torch.stack(box_list)
    box_batch_ids = torch.tensor(box_batch_idx, dtype=torch.long)
    proposals = torch.tensor(proposal_list, dtype=box_batch.dtype)
    proposal_batch_ids = torch.tensor(proposal_batch_idx, dtype=torch.long)
    assert len(box_batch) == len(box_batch_ids)
    assert len(proposals) == len(proposal_batch_ids)

    return img_batch, box_batch, box_batch_ids, proposals, proposal_batch_ids, w_batch, h_batch, img_id_list


def create_dataloader(dataset, batch_size, num_workers=0, shuffle=False, proposal_path=None):
    """
    Create data loader for SimpleVOCDataset.
    """
    collate_fn_partial = partial(collate_fn, proposal_path=proposal_path)
    loader = DataLoader(dataset,
                       batch_size=batch_size,
                       shuffle=shuffle,
                       pin_memory=True,
                       num_workers=num_workers,
                       collate_fn=collate_fn_partial)
    return loader


def parse_args():
    parser = argparse.ArgumentParser('Fast R-CNN Inference Checker', add_help=False)
    parser.add_argument('--data_base_dir', type=str, default='/data2/hw_data/hw2',
                        help='Base directory for the input data')
    parser.add_argument('--output_dir', type=str, default='./exp/inference_check_output',
                        help='Path to the output visualization directory')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='NMS IoU threshold')
    
    args = parser.parse_args()
    os.environ['TORCH_HOME'] = args.data_base_dir
    args.input_dir = os.path.join(args.data_base_dir, 'inference_check/input')
    args.checkpoint = os.path.join(args.data_base_dir, 'inference_check/checkpoint.pth')
    return args


def check_inference(args):
    """
    Run inference on the saved dataset and visualize the results.
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.input_dir}...")
    dataset = SimpleVOCDataset(args.input_dir)
    
    # Check if proposals exist
    proposal_path = os.path.join(args.input_dir, 'proposals')
    if not os.path.exists(proposal_path):
        raise FileNotFoundError(f"Proposals directory not found at {proposal_path}")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset, 
        args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        proposal_path=proposal_path
    )
    
    # Initialize model
    print("Initializing model...")
    model = FastRCNN()
    model.cuda()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Run inference
    model.eval()
    print("Running inference and generating visualizations...")
    print(f"Confidence threshold: {args.thresh}, NMS threshold: {args.nms_thresh}")
    
    total_images = 0
    total_gt_boxes = 0
    total_predictions = 0
    
    for iter_num, data_batch in enumerate(dataloader):
        images, boxes, boxes_batch_ids, proposals, proposal_batch_ids, w_batch, h_batch, img_ids = data_batch
        
        # Move data to GPU
        images = images.to(dtype=torch.float, device='cuda')
        resized_proposals = coord_trans(proposals, proposal_batch_ids, w_batch, h_batch, mode='p2a')
        resized_proposals = resized_proposals.to(dtype=torch.float, device='cuda')
        proposal_batch_ids = proposal_batch_ids.cuda()
        
        # Run inference
        with torch.no_grad():
            final_proposals, final_conf_scores, final_class = \
                model.inference(images, resized_proposals, proposal_batch_ids, 
                              thresh=args.thresh, nms_thresh=args.nms_thresh)
        
        # Process each image in the batch
        batch_size = len(images)
        for idx in range(batch_size):
            # Clamp proposal coordinates
            torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
            torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])
            
            # Get the original image
            i = batch_size * iter_num + idx
            img, _ = dataset.__getitem__(i)
            
            # Get ground truth boxes
            box_per_img = boxes[boxes_batch_ids == idx]
            
            # Prepare final predictions
            final_all = torch.cat(
                (final_proposals[idx], final_class[idx].float(), final_conf_scores[idx]), 
                dim=-1
            ).cpu()
            final_batch_idx = torch.LongTensor([idx] * final_all.shape[0])
            resized_final_proposals = coord_trans(final_all, final_batch_idx, w_batch, h_batch)
            
            # Visualize and save
            output_path = os.path.join(args.output_dir, img_ids[idx])
            data_visualizer(img, idx_to_class, output_path, box_per_img, resized_final_proposals)
            
            total_images += 1
            total_gt_boxes += len(box_per_img)
            total_predictions += resized_final_proposals.shape[0]
            
            print(f"✓ {img_ids[idx]}: {len(box_per_img)} GT boxes, "
                  f"{resized_final_proposals.shape[0]} predictions")
    
    print(f"\n{'='*60}")
    print(f"Inference check complete!")
    print(f"{'='*60}")
    print(f"Total images processed: {total_images}")
    print(f"Total GT boxes: {total_gt_boxes}")
    print(f"Total predictions: {total_predictions}")
    print(f"Average predictions per image: {total_predictions/total_images:.1f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = parse_args()
    check_inference(args)

