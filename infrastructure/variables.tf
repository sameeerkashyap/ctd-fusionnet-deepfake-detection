variable "project_id" {
  description = "GCP project ID where resources will be created."
  type        = string
}

variable "region" {
  description = "Region for regional resources."
  type        = string
}

variable "zone" {
  description = "Zone for the compute instance."
  type        = string
}

variable "instance_name" {
  description = "Name for the Deep Learning VM instance."
  type        = string
}

variable "machine_type" {
  description = "Machine type for the instance."
  type        = string
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GB."
  type        = number
}

variable "dl_image_family" {
  description = "Deep Learning VM image family (PyTorch + CUDA m131 release)."
  type        = string
}

variable "dl_image_project" {
  description = "GCP project that hosts the Deep Learning VM image."
  type        = string
}

variable "ssh_key_name" {
  description = "Base filename for the generated SSH key (stored under ../.ssh)."
  type        = string
}

variable "ssh_username" {
  description = "Linux username for SSH access (controls metadata entry and command)."
  type        = string
}

variable "metadata_startup_script" {
  description = "Optional startup script to run on instance boot."
  type        = string
  default     = ""
}

