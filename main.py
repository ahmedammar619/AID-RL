#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for the AID-RL project.
Provides a command-line interface to run the various components.
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime

# Import project modules
from data.db_config import DatabaseHandler
from models.rl_agent import ActorCriticAgent
from env.delivery_env import DeliveryEnv
from clustering.dbscan_cluster import RecipientClusterer
from training.train_agent import AgentTrainer
from feedback.feedback_handler import FeedbackHandler
from assignment.assign_volunteers import VolunteerAssigner


def setup_parser():
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(description='AID-RL: Volunteer Assignment Optimization with Reinforcement Learning')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Initialize database command
    init_parser = subparsers.add_parser('init-db', help='Initialize database tables')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the RL agent')
    train_parser.add_argument('--episodes', type=int, default=500, help='Number of episodes to train for')
    train_parser.add_argument('--steps', type=int, default=200, help='Maximum steps per episode')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    train_parser.add_argument('--device', type=str, default='auto', help='Device to run on (cpu, cuda, auto)')
    
    # Generate assignments command
    assign_parser = subparsers.add_parser('assign', help='Generate assignments using trained agent')
    assign_parser.add_argument('--agent', type=str, required=True, help='Path to trained agent checkpoint')
    assign_parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    assign_parser.add_argument('--visualize', action='store_true', help='Visualize assignments')
    assign_parser.add_argument('--save-report', action='store_true', help='Save assignment report')
    assign_parser.add_argument('--export-csv', action='store_true', help='Export assignments to CSV')
    
    # Run pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the complete pipeline')
    pipeline_parser.add_argument('--agent', type=str, required=True, help='Path to trained agent checkpoint')
    pipeline_parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output files')
    
    # Clustering command
    cluster_parser = subparsers.add_parser('map', help='Cluster recipients')
    cluster_parser.add_argument('--eps', type=float, default=0.00005, help='DBSCAN epsilon parameter')
    cluster_parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples parameter')
    cluster_parser.add_argument('--min-cluster-size', type=int, default=2, help='DBSCAN min_cluster_size parameter')
    
    # Feedback command
    feedback_parser = subparsers.add_parser('feedback', help='Handle admin feedback')
    feedback_parser.add_argument('--generate-report', action='store_true', help='Generate feedback report')
    feedback_parser.add_argument('--load', type=str, help='Load feedback from file')
    feedback_parser.add_argument('--save', type=str, help='Save feedback to file')
    
    return parser


# def initialize_database():
#     """Initialize the database tables."""
#     print("Initializing database...")
#     db = DatabaseHandler()
#     db.create_tables()
#     print("Database tables created successfully!")


def view_map(args):
    """View the recipient map."""
    
    print("Viewing recipient map...")

    # Get recipient coordinates
    db = DatabaseHandler()
    recipients = db.get_all_recipients()

    print(len(recipients))
    # Combine all points
    all_coords = np.array([[r.latitude, r.longitude] for r in recipients])
    
    # Create recipient IDs
    all_ids = [r.recipient_id for r in recipients]
    recipient_boxes = [r.num_items for r in recipients]
    
    # Get volunteer coordinates
    # Combine into a list of tuples
    volunteer_coords = np.array([[v.latitude, v.longitude] for v in db.get_all_volunteers()])
    # Initialize and fit the clusterer
    clusterer = RecipientClusterer(
        min_cluster_size=args.min_cluster_size,
        cluster_selection_epsilon=args.eps,
        min_samples=args.min_samples
    )
    labels = clusterer.fit(all_coords)

    # Get pickups
    pickups = db.get_all_pickups()
    pickup_coords = np.array([[p.latitude, p.longitude] for p in pickups])
    
    # Visualize the clusters
    output_path = os.path.join("./output", "cluster_map.html")
    clusterer.visualize_clusters(
        all_coords, 
        all_ids, 
        recipient_boxes,
        volunteer_coords,
        save_path=output_path,
        pickup_coords=pickup_coords
    )
    print("Map visualization complete!")

    

def train_agent(args):
    """Train the RL agent."""
    print("Training RL agent...")
    

    # Create database handler
    db_handler = DatabaseHandler()
    
    max_steps = 60

    # Create environment
    env = DeliveryEnv(db_handler=db_handler, max_steps=max_steps, use_clustering=True, cluster_eps=0.00005)
    
    # Create trainer
    trainer = AgentTrainer(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        db_handler=db_handler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        actor_lr=0.0001,
        critic_lr=0.0002,
        gamma=0.95
    )
    
    # Training loop
    stats = trainer.train(
        env=env,
        num_episodes=50,
        max_steps=max_steps,
        print_interval=10,
        checkpoint_interval=150,
        agent_num_updates=10
    )
    
    print("Training complete!")



# def generate_assignments(args):
#     """Generate assignments using a trained agent."""
#     print("Generating assignments...")
    
#     # Create assigner
#     assigner = VolunteerAssigner()
    
#     # Load agent
#     success = assigner.load_agent(args.agent)
#     if not success:
#         print("Failed to load agent!")
#         return
    
#     # Generate assignments
#     success = assigner.generate_assignments(deterministic=args.deterministic)
#     if not success:
#         print("Failed to generate assignments!")
#         return
    
#     # Save to database
#     assigner.save_assignments_to_db()
    
#     # Export to CSV if requested
#     if args.export_csv:
#         assigner.export_assignments_to_csv()
    
#     # Visualize if requested
#     if args.visualize:
#         assigner.visualize_assignments()
#         assigner.visualize_volunteer_load()
    
#     # Generate and save report if requested
#     if args.save_report:
#         report = assigner.generate_assignment_report()
#         assigner.save_report(report)
#         print("\nAssignment Report:")
#         print(report)
    
#     print("Assignment generation complete!")


def run_pipeline(args):
    """Run the complete assignment pipeline."""
    print("Running complete assignment pipeline...")
    
    # Create assigner
    assigner = VolunteerAssigner(output_dir=args.output_dir)
    
    # Run pipeline
    success = assigner.run_complete_pipeline(
        agent_path=args.agent,
        export_csv=True,
        save_visualizations=True,
        save_report=True
    )
    
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed!")


def handle_feedback(args):
    """Handle admin feedback."""
    print("Handling admin feedback...")
    
    # Create feedback handler
    handler = FeedbackHandler()
    
    # Load feedback if requested
    if args.load:
        success = handler.load_feedback(args.load)
        if success:
            print(f"Feedback loaded from {args.load}")
        else:
            print(f"Failed to load feedback from {args.load}")
    
    # Generate report if requested
    if args.generate_report:
        report = handler.generate_feedback_report()
        print("\nFeedback Report:")
        print(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("./output", f"feedback_report_{timestamp}.md")
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Report saved to {filepath}")
    
    # Save feedback if requested
    if args.save:
        filepath = handler.save_feedback(args.save)
        print(f"Feedback saved to {filepath}")
    
    print("Feedback handling complete!")


def main():
    """Main function to parse arguments and run commands."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == 'map':
        view_map(args)
    elif args.command == 'train':
        train_agent(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'feedback':
        handle_feedback(args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
