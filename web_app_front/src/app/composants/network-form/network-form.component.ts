import { Component, OnInit } from '@angular/core';
import { NetworkService } from '../../services/network.service';
import * as XLSX from 'xlsx';

@Component({
  selector: 'app-network-form',
  templateUrl: './network-form.component.html',
  styleUrl: './network-form.component.css'
})
export class NetworkFormComponent implements OnInit {

  response: any = null;

  selectedFile: File | null = null;

  constructor(private networkService: NetworkService) {}

  ngOnInit(): void {}

  onFileChange($event: any) {
    const files = $event.target.files;
    if (files.length) {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = (event: any) => {
            const wb = XLSX.read(event.target.result);
        }
        reader.readAsArrayBuffer(file);
        this.selectedFile = file;
        console.log('file = ', this.selectedFile);
    }
  }

  uploadFile() {
    if (this.selectedFile) {

      const formData = new FormData();
      formData.append('file', this.selectedFile);
      this.networkService.analyze(formData)
        .subscribe({
          next: (response) => {
            console.log('Succès :', response);
            this.response = response;
          },
          error: (error) => {
            console.error('Erreur lors de l\'envoi du fichier :', error);
          }
        });
    }
  }

}
