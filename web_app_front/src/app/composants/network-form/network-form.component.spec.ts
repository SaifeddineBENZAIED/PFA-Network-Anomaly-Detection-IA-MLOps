import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NetworkFormComponent } from './network-form.component';

describe('NetworkFormComponent', () => {
  let component: NetworkFormComponent;
  let fixture: ComponentFixture<NetworkFormComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [NetworkFormComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(NetworkFormComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
