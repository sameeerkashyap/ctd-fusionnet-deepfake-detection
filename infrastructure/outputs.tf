output "instance_self_link" {
  description = "Self-link of the Deep Learning VM instance."
  value       = google_compute_instance.dlvm.self_link
}

output "instance_name" {
  description = "Name of the created instance."
  value       = google_compute_instance.dlvm.name
}

output "instance_ip" {
  description = "External IP address for SSH connections."
  value       = google_compute_instance.dlvm.network_interface[0].access_config[0].nat_ip
}

output "ssh_private_key_path" {
  description = "Path to the generated private SSH key."
  value       = local_file.private_key.filename
  sensitive   = true
}

output "ssh_command" {
  description = "SSH command to access the instance using the generated key."
  value       = format(
    "ssh -i %s %s@%s",
    local_file.private_key.filename,
    var.ssh_username,
    google_compute_instance.dlvm.network_interface[0].access_config[0].nat_ip
  )
}

output "bucket_name" {
  description = "Cloud Storage bucket created for the workload."
  value       = google_storage_bucket.artifact_bucket.name
}

