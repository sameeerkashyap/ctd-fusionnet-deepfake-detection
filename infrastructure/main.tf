provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

data "google_compute_image" "deep_learning_pytorch" {
  project = var.dl_image_project
  family  = var.dl_image_family
}

resource "tls_private_key" "ssh" {
  algorithm = "ED25519"
}

locals {
  ssh_dir        = "${path.module}/../.ssh"
  private_key    = "${local.ssh_dir}/${var.ssh_key_name}"
  public_key     = "${local.private_key}.pub"
  ssh_key_string = "${var.ssh_username}:${tls_private_key.ssh.public_key_openssh}"
}

resource "local_file" "private_key" {
  content              = tls_private_key.ssh.private_key_openssh
  filename             = local.private_key
  directory_permission = "0700"
  file_permission      = "0600"
}

resource "local_file" "public_key" {
  content              = tls_private_key.ssh.public_key_openssh
  filename             = local.public_key
  directory_permission = "0700"
  file_permission      = "0644"
}

resource "google_compute_instance" "dlvm" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  tags = [
    "fusionnet",
    "deep-learning",
    "pytorch"
  ]

  boot_disk {
    initialize_params {
      image = data.google_compute_image.deep_learning_pytorch.self_link
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
    auto_delete       = true
    kms_key_self_link = null
  }

  network_interface {
    network = "default"

    access_config {
      // Ephemeral public IP
    }
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    provisioning_model  = "STANDARD"
  }

  metadata = {
    "ssh-keys"              = local.ssh_key_string
    "install-nvidia-driver" = "true"
    "proxy-mode"            = "project_editors"
    "serial-port-enable"    = "true"
    "startup-script"        = var.metadata_startup_script
    "enable-oslogin"        = "FALSE"
  }

  labels = {
    workload = "deep-learning"
    runtime  = "pytorch"
  }

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  advanced_machine_features {
    threads_per_core = 2
  }
}

