import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { NetworkFormComponent } from './composants/network-form/network-form.component';

const routes: Routes = [
  { path: '', component: NetworkFormComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
