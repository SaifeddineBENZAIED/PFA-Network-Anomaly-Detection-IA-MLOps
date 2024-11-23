import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class NetworkService {

  private apiUrl = 'http://localhost:8000/simulation/simulate/'; // URL du backend Django
  private apiUrl2 = 'http://localhost:8000/simulation/create/'; // URL du backend Django

  constructor(private http: HttpClient) {}

  simulateNetwork(data: any): Observable<any> {
    return this.http.post(this.apiUrl, data);
  }

  createNetwork(data: any): Observable<any> {
    return this.http.post(this.apiUrl2, data);
  }

  analyze(formData: any): Observable<any> {
    return this.http.post('http://localhost:8000/simulation/analyze/', formData);
  }

}
